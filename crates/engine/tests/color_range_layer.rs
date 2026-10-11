//! Select › Color Range from one layer: the layer's own colours, the parts other layers cover
//! included; with Sample All Layers, the composite.

use photocraft_engine::Session;
use serde_json::json;

fn selected(s: &Session, x: i32, y: i32) -> f32 {
    s.active().unwrap().doc.selection.as_ref().map_or(0.0, |sel| sel.sample_channel(x, y, 0))
}

#[test]
fn color_range_samples_one_layer_or_the_image() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 60, "height": 20, "background": "white"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "red"})).unwrap();
    let red = s.active().unwrap().active_layer.unwrap();
    s.execute("paint.stroke", json!({"points": [[0, 10], [60, 10]], "size": 30, "hardness": 1.0, "color": "#ff0000"})).unwrap();
    // A blue layer above covers the right half.
    s.execute("layer.new.layer", json!({"name": "blue"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[45, 10], [60, 10]], "size": 40, "hardness": 1.0, "color": "#0000ff"})).unwrap();
    s.execute("layer.select", json!({"layer": red.0})).unwrap();
    s.execute("select.colorRange", json!({"color": "#ff0000", "fuzziness": 20, "sampleAllLayers": false})).unwrap();
    assert!(selected(&s, 10, 10) > 0.9);
    assert!(selected(&s, 55, 10) > 0.9, "the red under the blue layer is selected too");
    s.execute("select.colorRange", json!({"color": "#ff0000", "fuzziness": 20, "sampleAllLayers": true})).unwrap();
    assert!(selected(&s, 10, 10) > 0.9);
    assert!(selected(&s, 55, 10) < 0.1, "the image shows blue there");
    // A Smart Object samples its own rendering.
    s.execute("layer.smartObjects.convertToSmartObject", json!({"layer": red.0})).unwrap();
    s.execute("select.colorRange", json!({"color": "#ff0000", "fuzziness": 20, "sampleAllLayers": false})).unwrap();
    assert!(selected(&s, 55, 10) > 0.9, "smart object alone");
}
