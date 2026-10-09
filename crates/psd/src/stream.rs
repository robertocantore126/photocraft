//! Native streaming parser: read channel payloads per layer, without a whole-file byte copy.
use crate::io::Reader;
use crate::{Compression, GlobalLayerMask, Header, LayerInfo, LayerInfoPlacement, LayerRecord, PsdError, PsdFile, Result};
use std::io::{Read, Seek};

struct Input<'a, R> {
    reader: &'a mut R,
    position: u64,
    end: u64,
}
impl<R: Read + Seek> Input<'_, R> {
    fn bytes(&mut self, length: u64) -> Result<Vec<u8>> {
        let next = self.position.checked_add(length).ok_or(PsdError::LimitExceeded("stream offset overflow"))?;
        if next > self.end {
            return Err(PsdError::invalid("stream section truncated"));
        }
        let size = usize::try_from(length).map_err(|_| PsdError::LimitExceeded("stream length exceeds address space"))?;
        if length > 1 << 31 {
            return Err(PsdError::LimitExceeded("stream allocation exceeds 2 GiB"));
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size).map_err(|_| PsdError::LimitExceeded("stream allocation failed"))?;
        bytes.resize(size, 0);
        self.reader.read_exact(&mut bytes).map_err(|e| PsdError::invalid(format!("stream read: {e}")))?;
        self.position = next;
        Ok(bytes)
    }
    fn peek(&mut self, length: u64) -> Result<Vec<u8>> {
        let start = self.position;
        let bytes = self.bytes(length.min(self.end.saturating_sub(start)))?;
        self.reader.seek(std::io::SeekFrom::Start(start)).map_err(|e| PsdError::invalid(e.to_string()))?;
        self.position = start;
        Ok(bytes)
    }
    fn length(&mut self, long: bool) -> Result<u64> {
        let bytes = self.bytes(if long { 8 } else { 4 })?;
        let mut r = Reader::new(&bytes);
        r.len_field(long)
    }
    fn layers(&mut self, length: u64, version: crate::Version) -> Result<LayerInfo> {
        let start = self.position;
        let outer_end = self.end;
        let end = start.checked_add(length).filter(|end| *end <= outer_end).ok_or_else(|| PsdError::invalid("layer section truncated"))?;
        self.end = end;
        let count_bytes = self.bytes(2)?;
        let count = Reader::new(&count_bytes).i16()?;
        let count_abs = usize::from(count.unsigned_abs());
        if (count_abs as u64).saturating_mul(34) > self.end.saturating_sub(self.position) {
            return Err(PsdError::invalid("layer count exceeds section"));
        }
        let mut layers = Vec::with_capacity(count_abs);
        let mut channel_lengths = Vec::with_capacity(count_abs);
        for _ in 0..count_abs {
            let mut record = self.bytes(18)?;
            let mut prefix = Reader::new(&record);
            prefix.skip(16)?;
            let channels = usize::from(prefix.u16()?);
            if channels > crate::layer::MAX_LAYER_CHANNELS as usize {
                return Err(PsdError::LimitExceeded("too many layer channels"));
            }
            let stride: usize = if version.is_psb() { 10 } else { 6 };
            let tail_size = channels.checked_mul(stride).and_then(|n| n.checked_add(16)).ok_or(PsdError::LimitExceeded("record length overflow"))?;
            record.extend(self.bytes(tail_size as u64)?);
            let at = record.len().checked_sub(4).ok_or_else(|| PsdError::invalid("record truncated"))?;
            let extra = Reader::new(record.get(at..).ok_or_else(|| PsdError::invalid("record truncated"))?).u32()?;
            record.extend(self.bytes(u64::from(extra))?);
            let (layer, lengths) = LayerRecord::read(&mut Reader::new(&record), version)?;
            layers.push(layer);
            channel_lengths.push(lengths);
        }
        for (layer, lengths) in layers.iter_mut().zip(channel_lengths) {
            for (channel, length) in layer.channels.iter_mut().zip(lengths) {
                if length == 0 {
                    continue;
                }
                if length < 2 {
                    return Err(PsdError::invalid("channel data length 1"));
                }
                let tag = self.bytes(2)?;
                channel.compression = Some(Compression::from_u16(Reader::new(&tag).u16()?));
                channel.data = self.bytes(length - 2)?;
            }
        }
        let body_length = self.position - start;
        let rest = self.bytes(end - self.position)?;
        let padding = if rest.len() as u64 == body_length % 2 && rest.iter().all(|&b| b == 0) { None } else { Some(rest) };
        self.end = outer_end;
        Ok(LayerInfo { merged_alpha: count < 0, layers, padding })
    }
}

impl PsdFile {
    /// Parse a seekable native source. Encoded channels are read individually in file order;
    /// the format-level model still retains those channels for lossless export.
    pub fn from_reader(reader: &mut (impl Read + Seek)) -> Result<Self> {
        let start = reader.stream_position().map_err(|e| PsdError::invalid(e.to_string()))?;
        let end = reader.seek(std::io::SeekFrom::End(0)).map_err(|e| PsdError::invalid(e.to_string()))?;
        reader.seek(std::io::SeekFrom::Start(start)).map_err(|e| PsdError::invalid(e.to_string()))?;
        let mut input = Input { reader, position: start, end };
        let header = Header::read(&mut Reader::new(&input.bytes(26)?))?;
        let psb = header.version.is_psb();
        let color_len = input.length(false)?;
        let color_mode_data = input.bytes(color_len)?;
        let resource_len = input.length(false)?;
        let mut resource_bytes = Vec::new();
        resource_bytes.extend((resource_len as u32).to_be_bytes());
        resource_bytes.extend(input.bytes(resource_len)?);
        let resources = crate::resources::read_section(&mut Reader::new(&resource_bytes))?;
        drop(resource_bytes);
        let layer_mask_length = input.length(psb)?;
        let layer_mask_end =
            input.position.checked_add(layer_mask_length).filter(|end| *end <= input.end).ok_or_else(|| PsdError::invalid("layer/mask section truncated"))?;
        let file_end = input.end;
        input.end = layer_mask_end;
        let mut layer_info = None;
        let mut global_layer_mask = None;
        let mut global_blocks = Vec::new();
        let mut layer_mask_trailing = Vec::new();
        let mut layer_info_placement = LayerInfoPlacement::Section;
        if layer_mask_length > 0 {
            let layer_length = input.length(psb)?;
            if layer_length > 0 {
                layer_info = Some(input.layers(layer_length, header.version)?);
            }
            if input.end.saturating_sub(input.position) >= 4 {
                let mask_length = input.length(false)?;
                global_layer_mask = Some(GlobalLayerMask { data: input.bytes(mask_length)? });
                while input.end.saturating_sub(input.position) >= 12 {
                    let prefix = input.peek(8)?;
                    if !(prefix.starts_with(b"8BIM") || prefix.starts_with(b"8B64")) {
                        break;
                    }
                    let header_bytes = input.bytes(8)?;
                    let mut block_header = Reader::new(&header_bytes);
                    let signature = block_header.array::<4>()?;
                    let key = block_header.array::<4>()?;
                    let length = input.length(crate::tagged::uses_long_length(header.version, &key))?;
                    let start = input.position;
                    let section_end = input.end;
                    let decoded = if layer_info.is_none() && matches!(&key, b"Lr16" | b"Lr32" | b"Layr") {
                        match input.layers(length, header.version) {
                            Ok(info) => Some(info),
                            Err(_) => {
                                input.reader.seek(std::io::SeekFrom::Start(start)).map_err(|e| PsdError::invalid(e.to_string()))?;
                                input.position = start;
                                input.end = section_end;
                                None
                            }
                        }
                    } else {
                        None
                    };
                    let data = if decoded.is_some() { Vec::new() } else { input.bytes(length)? };
                    let rest = input.peek(7)?;
                    let padding_length = (0..=3usize)
                        .find(|&k| {
                            k == rest.len() && input.position + k as u64 == input.end
                                || rest.get(k..).is_some_and(|b| b.starts_with(b"8BIM") || b.starts_with(b"8B64"))
                        })
                        .unwrap_or(0);
                    let pad = input.bytes(padding_length as u64)?;
                    let padding = if pad.len() as u64 == length % 2 && pad.iter().all(|&b| b == 0) { None } else { Some(pad) };
                    if let Some(info) = decoded {
                        layer_info = Some(info);
                        layer_info_placement = LayerInfoPlacement::GlobalBlock { index: global_blocks.len(), signature, key, padding };
                    } else {
                        global_blocks.push(crate::TaggedBlock { signature, key, data, padding });
                    }
                }
                layer_mask_trailing = input.bytes(input.end - input.position)?;
            } else {
                layer_mask_trailing = input.bytes(input.end - input.position)?;
            }
        }
        input.end = file_end;
        let compression = Compression::from_u16(Reader::new(&input.bytes(2)?).u16()?);
        let data = input.bytes(file_end - input.position)?;
        crate::compression::validate_planes(compression, &data, &crate::ImageData::layout(&header))?;
        let image_data = crate::ImageData { compression, data };
        Ok(PsdFile { header, color_mode_data, resources, layer_info, layer_info_placement, global_layer_mask, global_blocks, layer_mask_trailing, image_data })
    }
}
