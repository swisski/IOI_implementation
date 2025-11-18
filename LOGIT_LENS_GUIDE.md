# Logit Lens Analysis Guide

## What is Logit Lens?

Logit Lens is a technique for understanding **how neural networks build up their predictions layer by layer**. Instead of only looking at the final output, we "peek" at intermediate layers to see what the model "thinks" at each stage.

## How It Works

For the IOI (Indirect Object Identification) task:
- **Input**: "When Alice and Bob went to the store, Alice gave a bottle to ___"
- **Correct Answer**: Bob (the indirect object / IO token)
- **Incorrect Answer**: Alice (the subject / S token that appeared twice)

At each layer, we:
1. Extract the residual stream (the model's internal representation)
2. Project it through the unembedding matrix (W_U) to get logits
3. Compute the logit difference: `logit(Bob) - logit(Alice)`
4. Track how this difference evolves

**Higher logit diff = Stronger preference for the correct answer (Bob)**

## Example Results

```
Layer-wise Logit Difference:
  Embed :   0.333   ← Initial embeddings: weak preference
  L0    :   9.801   ← Duplicate token heads boost signal
  L1-6  :   9-16    ← Gradual build-up
  L7    :  23.609   ← S-inhibition heads strengthen signal
  L8    :  33.896
  L9    :  76.754   ← Name mover heads DOMINATE
  L10   :  90.078   ← Peak preference
  Final :   3.233   ← After layernorm (preference maintained)
```

### Key Insights:

1. **Early layers (L0-3)**: Duplicate token heads provide first major boost (+9.5)
2. **Middle layers (L7-8)**: S-inhibition heads accelerate preference
3. **Late layers (L9-10)**: Name mover heads create explosive growth (+57!)
4. **Final layer**: Layernorm normalizes but maintains relative preference

## Usage

### Quick Start

```python
from src.analysis.logit_lens import compute_layer_wise_logit_diff, plot_logit_lens
from src.model.model_loader import load_ioi_model

# Load model
result = load_ioi_model(device="cuda")
model = result["model"]

# Prepare tokens
prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
tokens = model.to_tokens(prompt)
bob_id = model.to_single_token(" Bob")
alice_id = model.to_single_token(" Alice")

# Compute logit lens
results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)

# Visualize
plot_logit_lens(results, save_path="my_logit_lens.png")
```

### Analyze Across Dataset

```python
from src.analysis.logit_lens import analyze_logit_lens_for_dataset

# Average across 100 examples
results = analyze_logit_lens_for_dataset(
    model,
    "data/ioi_abba.json",
    max_examples=100
)

# Results contain:
# - mean_logit_diffs: Average at each layer
# - std_logit_diffs: Standard deviation
# - all_logit_diffs: Individual results (examples × layers)

print(f"Layer 0: {results['mean_logit_diffs'][0]:.3f}")
print(f"Layer 11: {results['mean_logit_diffs'][-1]:.3f}")
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Get results
lens_avg = analyze_logit_lens_for_dataset(model, "data/ioi_abba.json", max_examples=50)

# Create custom plot
fig, ax = plt.subplots(figsize=(14, 6))
layers = np.arange(len(lens_avg['mean_logit_diffs']))

# Plot with error bars
ax.errorbar(layers, lens_avg['mean_logit_diffs'],
            yerr=lens_avg['std_logit_diffs'],
            fmt='o-', linewidth=2, markersize=8, capsize=5)

# Add circuit component regions
ax.axvspan(-0.5, 3.5, alpha=0.1, color='purple', label='Duplicate Token (L0-3)')
ax.axvspan(6.5, 8.5, alpha=0.1, color='orange', label='S-Inhibition (L7-8)')
ax.axvspan(8.5, 11.5, alpha=0.1, color='green', label='Name Mover (L9-11)')

ax.set_xlabel('Layer')
ax.set_ylabel('Logit Difference (IO - S)')
ax.set_title('Logit Lens: How Model Builds Up Answer')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('custom_logit_lens.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Interpretation Guide

### What to Look For

1. **Initial State (Embed)**:
   - Should be close to 0 or slightly positive
   - Shows model's prior bias before computation

2. **Early Layers (L0-3)**:
   - First major increase should occur here
   - Indicates duplicate token heads identifying repeated names

3. **Middle Layers (L7-8)**:
   - Second acceleration phase
   - Shows S-inhibition heads working

4. **Late Layers (L9-11)**:
   - Largest increases here
   - Name mover heads provide final strong signal

5. **Layer Transitions**:
   - Big jumps indicate important computations
   - Analyze `np.diff(results['layer_logit_diffs'])` to find key transitions

### Red Flags

- **No increase through layers**: Circuit not working
- **Decrease in middle layers**: S-inhibition might be backwards
- **Flat at late layers**: Name movers not engaging
- **High variance across examples**: Noisy/inconsistent behavior

## Comparison with Paper

The IOI paper (Wang et al., 2022) proposed a circuit with three main components:
1. Duplicate token heads (early)
2. S-inhibition heads (middle)
3. Name mover heads (late)

**Logit lens validates this proposal** by showing:
- ✅ Preference builds up in stages
- ✅ Each stage corresponds to proposed component layer ranges
- ✅ Largest effect from name movers (as paper claims)
- ✅ Clear specialization by layer

## Advanced Analysis

### Find Most Important Layer Transitions

```python
deltas = np.diff(results['mean_logit_diffs'])
top_transitions = np.argsort(deltas)[::-1][:5]

print("Top 5 most important layer transitions:")
for i, trans in enumerate(top_transitions, 1):
    print(f"{i}. Layer {trans-1} → {trans}: +{deltas[trans]:.3f}")
```

### Correlation with Circuit Heads

```python
from src.analysis.attention_analysis import find_all_ioi_heads

# Find circuit heads
circuit = find_all_ioi_heads(model, "data/ioi_abba.json", max_examples=50)

# Check if layers with circuit heads show larger increases
name_mover_layers = set(head[0] for head in circuit['name_mover_heads'])

for layer in name_mover_layers:
    delta = deltas[layer]
    print(f"Layer {layer} (has name mover): +{delta:.3f}")
```

### Variance Analysis

```python
# High variance = inconsistent behavior
variance = np.var(results['all_logit_diffs'], axis=0)

print("Layers with highest variance:")
high_var_layers = np.argsort(variance)[::-1][:3]
for layer in high_var_layers:
    print(f"  Layer {layer}: variance = {variance[layer]:.3f}")
```

## Integration with Notebook

Logit lens is integrated as **Phase 8** in the validation notebook:

```
notebooks/ioi_replication_validation.ipynb
  ...
  Phase 7: Direct Logit Attribution
  Phase 8: Logit Lens Analysis  ← NEW!
    - Single example analysis
    - Dataset average (100 examples)
    - Validation check for layer-wise build-up
    - Identifies top layer transitions
  ...
```

Run the notebook to see logit lens in action with beautiful visualizations!

## FAQ

**Q: Why does logit diff decrease at the final layer?**
A: The final layernorm normalizes activations, reducing absolute magnitudes. The relative preference (IO > S) is maintained.

**Q: Can I use this on other tasks?**
A: Yes! Just replace `io_token_id` and `s_token_id` with your target tokens. Works for any classification task.

**Q: Why do some layers show negative deltas?**
A: Some heads may suppress the signal temporarily. This can be part of the computation - not all layers must increase preference.

**Q: How many examples should I average over?**
A: 50-100 examples gives stable averages. Use more if high variance.

**Q: What if my logit lens is flat?**
A: Either:
- Circuit isn't working (check attention patterns)
- Wrong token IDs (check your io/s token extraction)
- Model hasn't learned the task (check baseline accuracy)

## References

- nostalgebraist (2020). "interpreting GPT: the logit lens"
- Wang et al. (2022). "Interpretability in the Wild: IOI in GPT-2 small"
- TransformerLens: https://github.com/neelnanda-io/TransformerLens

## Visualization Tips

1. **Always include error bars** when averaging over examples
2. **Shade circuit component regions** to show correspondence
3. **Use consistent y-axis scales** when comparing multiple prompts
4. **Add layer transition plot** to show where changes occur
5. **Annotate key layers** (e.g., "Name movers activate here")

---

**Happy analyzing!** 🔬 The logit lens reveals how transformers truly think layer by layer.
