<img width="323" height="275" alt="plus" src="https://github.com/user-attachments/assets/145146d0-b8b8-42fa-bc57-7e6f8879229c" />

### **Key Improvements:**
1.  **Added `seed`**: Now the random weights of the MLP and Attention layers are deterministic. You can reproduce your results.
2.  **Exposed `attn_strength`**: No longer hardcoded to 0.3. You can control how much the tokens "talk" to each other.
3.  **Added `clamp_range`**: Prevents the "rainbow artifacts" mentioned in the README by capping extreme values in the embedding.
4.  **Added `noise_injection`**: Adds a tiny amount of controlled noise *before* processing to help break specific stuck patterns in Qwen models.
5.  **Selectable Activation**: Choose between GELU, ReLU, or SiLU for different "flavors" of distortion.

### **What changes for the user?**

1.  **Seed Control:** You can now connect a `Primitive` or standard `Seed` node to this. If you find a "Magic Enhancement" that creates amazing textures, you can keep the seed fixed while changing your prompt, and the *style of distortion* will remain consistent.
2.  **Cleaner High Strength:** By using `clamp_output` (try setting it to `20.0` or `50.0`), you can push `enhance_strength` higher without the image turning into RGB static.
3.  **Different "Moods":**
    *   **GELU:** Smooth, modern AI look (Default).
    *   **ReLU:** Sharper, harsher cutoffs in the embeddings.
    *   **Tanh:** Softer, more "analog" saturation in the conditioning.
4.  **Dithering:** The `noise_injection` parameter (try `0.05`) is excellent for Qwen. Large Qwen embeddings sometimes "collapse" into uniform patterns. Injecting noise prevents the sampler from getting stuck in local minima.
