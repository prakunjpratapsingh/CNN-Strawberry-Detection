# CNN-Strawberry-Detection

Developed a CNN architecture for strawberry detection by truncating pre-trained model "MobileNetV3Large" at block 14 and adding a custom encoder architecture to generate heatmaps for precise strawberry localization, addressing challenges such as lighting variability, partial obstructions, and background clutter.



Strawberry Low Resolution Heatmap Detector (Encoder only) - MobileNetV3 (TensorFlow/Keras)


Implemented a low-resolution heatmap detector for ripe strawberries via transfer learning on pretrained model MobileNetV3-Large with encoder output at 1/32 resolution.


Truncated at layer expanded_conv_12_project_bn (10×40×160).


Implemented custom block on top of truncation layer 1×1 expand (160→16) → 3×3 depthwise (16 dw) → 1×1 project (16→4) (+ BN & ReLU).


Final 1×1 conv → sigmoid to produce 1 channel heatmap.


Backbone frozen for head training (10 epochs) then unfrozen for Full fine-tune (5 epochs)


Results (val): loss ≈ 0.061, IoU@0.5 ≈ 0.073, precision ≈ 0.28, recall ≈ 0.09 after backbone fine-tuning; qualitative improvements in localization vs. head-only training.


Trained Head-only (10 epochs): val loss : 0.078–0.080 and then Full fine-tune (5 epochs): val loss ≈ 0.061, IoU@0.5 ≈ 0.073, precision ≈ 0.28, recall ≈ 0.09.



Strawberry High Resolution Heatmap Detector: Encoder decoder





Designed a 3-stage decoder (nearest-neighbor upsample → squeeze → concat → MBConv-style our_block with depthwise 3×3, expansion=2), followed by a refine 3×3 (4 ch) + 4× upsample to full resolution.


Omitted the highest-res skip to avoid misaligned padding/aliasing and to stay within the memory envelope; kept all high-res tensors <27 MB (largest: 320×1280×4 ≈ 6.6 MB).


Ultra-lean head: depthwise 3×3 → 1×1 → sigmoid producing a full-res heatmap (320×1280×1); logits initialized with class-prior bias.


Decoder trainables ≈2.2k params (total model 2.84 M; backbone mostly frozen initially), satisfying One-in-Ten parameter budget.


Achieved robust high-res detection: after full fine-tune, loss 0.0448, precision@0.1 ≈ 0.69, recall@0.1 ≈ 0.76 .
