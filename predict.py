import torch
import numpy as np
import pandas as pd

from models.model import AcDispNetL4


# Configuration parameter
GRID_SIZE = 50
FREQ_CLASSES = 31 # Frequency category
OUTPUT_FILE = 'results/predict/displacement_field' + str(GRID_SIZE) + '.csv'

# Generate displacement predictions for all frequencies and coordinate points
def generate_displacements():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)

    # Initialize the model and load the weights
    model = AcDispNetL4(3, 2)
    model.load_state_dict(torch.load('checkpoints/model_final_1.pth', weights_only=True))
    model.to(device)
    model.eval()

    # Generate a normalized coordinate grid
    x = torch.linspace(0.5 / GRID_SIZE, 1 - 0.5 / GRID_SIZE, GRID_SIZE)
    y = torch.linspace(0.5 / GRID_SIZE, 1 - 0.5 / GRID_SIZE, GRID_SIZE)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')  # Create grid

    # Actual coordinates of all grids
    coord_x = np.round(grid_x.numpy().flatten() * GRID_SIZE, decimals=3)
    coord_y = np.round(grid_y.numpy().flatten() * GRID_SIZE, decimals=3)

    # Initialize the lag for CSV writer
    first_write = True

    normalized_values = [i / (FREQ_CLASSES - 1) for i in range(FREQ_CLASSES)]

    # Traverse all frequency classes
    for freq_idx, value in enumerate(normalized_values):

        class_channel = torch.full((GRID_SIZE, GRID_SIZE), value)
        inputs = torch.stack([class_channel, grid_x, grid_y], dim=-1).unsqueeze(0)
        inputs = inputs.to(device)

        # Use model to predict
        with torch.no_grad():
            outputs = model(inputs)

        # Fetch results
        dx = outputs[0, :, :, 0].cpu().numpy().flatten()
        dy = outputs[0, :, :, 1].cpu().numpy().flatten()


        # Build the DataFrame for the current class
        df = pd.DataFrame({
            'frequency': np.full(GRID_SIZE * GRID_SIZE, freq_idx),
            'x': coord_x,
            'y': coord_y,
            'dx': dx,
            'dy': dy
        })

        # Write to CSV in batches to avoid memory overflow
        df.to_csv(
            OUTPUT_FILE,
            mode='w' if first_write else 'a',  # Creat file for the first time and append later
            header=first_write,
            index=False
        )
        first_write = False


if __name__ == "__main__":
    generate_displacements()
    print(f"Prediction results have been saved to {OUTPUT_FILE}")