import numpy as np
import matplotlib.pyplot as plt


def get_top_fingerprint_weights(model, top_k=20):
    """
    Extract top important fingerprint bits based on first MLP layer weights.
    
    Args:
        model: Trained BindingMLP model.
        top_k: Number of top bits to return.

    Returns:
        top_indices: Indices of top fingerprint bits.
        top_values: Corresponding importance scores.
    """
    first_layer = model.network[0]  # First linear layer
    weights = first_layer.weight.data.cpu().numpy()
    
    # Molecule features assumed to be last 2048 dims
    mol_weights = weights[:, 480:]  # Adjust 480 if protein feature dim changes
    
    bit_importance = np.sum(np.abs(mol_weights), axis=0)
    top_indices = np.argsort(bit_importance)[-top_k:][::-1]
    top_values = bit_importance[top_indices]
    
    return top_indices, top_values


def plot_top_fingerprint_bits(top_indices, top_values):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_indices)), top_values, tick_label=top_indices)
    plt.xlabel('Fingerprint Bit Index')
    plt.ylabel('Importance Score (Sum of Absolute Weights)')
    plt.title('Top Important Molecular Fingerprint Bits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
