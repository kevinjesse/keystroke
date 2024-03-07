import pandas as pd
import torch
from torch.utils.data import Dataset
from coatnet import (
    CoAtNet,
)  # Ensure this is correctly installed or accessible in your environment.
import torch.optim as optim
import torchaudio
from accelerate import Accelerator
import random
import numpy as np
import librosa

from accelerate.utils import (
    tqdm,
)  # Import tqdm from the accelerate package for progress bars.

batch_size = 256

# CoAtNet parameters
num_blocks = [
    2,
    2,
    12,
    28,
    2,
]  # Defines the number of blocks in each stage of the CoAtNet.
channels = [
    192,
    192,
    384,
    768,
    1536,
]  # Defines the channel dimensions for each stage of the CoAtNet.


def make_melspec(
    x: np.ndarray,
    n_mels: int = 64,
    sample_rate: int = 44100,
    hop_length: int = 225,
    n_fft: int = 1023,
) -> np.ndarray:
    """
    Convert an audio waveform into a Mel spectrogram.

    Parameters:
        x (np.ndarray): Input audio waveform.
        n_mels (int): Number of Mel bands to generate.
        sample_rate (int): Sampling rate of the audio waveform.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): Length of the FFT window.

    Returns:
        np.ndarray: The resulting Mel spectrogram.
    """
    S = librosa.feature.melspectrogram(
        y=x.squeeze(), sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_DB = librosa.power_to_db(S, ref=np.max)
    offset = np.abs(np.min(S_DB))
    S_DB += offset
    return S_DB


def SpecAugment(wav: np.ndarray) -> np.ndarray:
    """
    Apply SpecAugment on an audio waveform, including time shifting and Mel spectrogram masking.

    Parameters:
        wav (np.ndarray): Input audio waveform.

    Returns:
        np.ndarray: The augmented Mel spectrogram.
    """

    def time_shift(aud: np.ndarray, shift_limit: float) -> np.ndarray:
        _, sig_len = aud.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return np.roll(aud, shift_amt)

    def specaugment(
        mel: np.ndarray, n_mels: int = 64, max_mask_pct: float = 0.1, n_steps: int = 64
    ) -> torch.Tensor:
        time_mask_param = max_mask_pct * n_steps
        time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param
        )
        freq_mask_param = max_mask_pct * n_mels
        freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param
        )
        masked = time_masking(freq_masking(torch.tensor(mel))[None, :])
        return masked

    shift_aud = time_shift(wav, 0.4)
    mel = make_melspec(
        shift_aud, n_mels=64, sample_rate=44100, n_fft=1023, hop_length=225
    )
    masked = specaugment(mel, max_mask_pct=0.1)
    return masked.squeeze()


def NoAugment(wav: np.ndarray) -> np.ndarray:
    """
    Generate a Mel spectrogram from an audio waveform without applying augmentation.

    Parameters:
        wav (np.ndarray): Input audio waveform.

    Returns:
        np.ndarray: The Mel spectrogram.
    """
    mel = make_melspec(wav, n_mels=64, sample_rate=44100, n_fft=1023, hop_length=225)
    return mel.squeeze()


class MelDataset(Dataset):
    """
    A dataset class for Mel spectrograms, supporting augmentation transformations.

    Attributes:
        annotations_file (str or pd.DataFrame): Path to the annotations file or a DataFrame.
        transform (callable, optional): A function/transform that takes in an audio waveform and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in a target and transforms it.
        seed (int): Random seed for shuffling data.
    """

    def __init__(
        self, annotations_file, transform=None, target_transform=None, seed=10
    ):
        if isinstance(annotations_file, str):
            self.data = pd.read_pickle(annotations_file)
        elif isinstance(annotations_file, pd.DataFrame):
            self.data = annotations_file

        self.data = self.data.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )  # Shuffle the dataset.

        self.images = list(self.data.waveform)
        self.labels = list(self.data.label)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its label from the dataset at the specified index.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed sample and its label.
        """
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image[None, :], label


def run():
    """
    Main function to run the training and testing of the CoAtNet model on Mel spectrogram dataset.
    """
    # Initialization of datasets with and without augmentation
    train_dataset = MelDataset("data_webex.pkl", transform=SpecAugment)
    test_dataset = MelDataset("data_webex_holdout.pkl", transform=NoAugment)

    epochs = 200  # Number of training epochs

    def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        data_loader,
        device="cuda",
    ) -> int:
        """
        Train the model for one epoch.

        Parameters:
            model (torch.nn.Module): The neural network model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler: The LR scheduler.
            data_loader (DataLoader): The DataLoader for the training data.
            device (str): The device to use for training.

        Returns:
            int: The training accuracy percentage.
        """
        model = model.to(device)
        max_grad_norm = 1.0
        running_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            outputs = accelerator.gather(outputs)
            _, predicted = torch.max(outputs.data, 1)
            labels = accelerator.gather(labels).cpu()
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

        train_acc = 100 * correct // total
        return train_acc

    def eval(model: torch.nn.Module, data_loader, device="cuda") -> int:
        """
        Test the model on the test dataset.

        Parameters:
            model (torch.nn.Module): The neural network model to test.
            data_loader (DataLoader): The DataLoader for the test data.
            device (str): The device to use for testing.

        Returns:
            int: The testing accuracy percentage.
        """
        model = model.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                outputs = model(images.to(device))
                outputs = accelerator.gather(outputs)
                _, predicted = torch.max(outputs.data, 1)
                labels = accelerator.gather(labels).cpu()
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()

        return 100 * correct // total

    data = {"epoch": [], "test_acc": [], "train_acc": []}

    accelerator = Accelerator()
    device = accelerator.device

    # Model initialization
    model = CoAtNet((64, 64), 1, num_blocks, channels, num_classes=36)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=10)

    # Data loaders for training and testing
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=min(len(test_dataset), batch_size),
        shuffle=True,
        drop_last=False,
    )

    # Preparing model, optimizer, scheduler, and dataloaders for distributed training
    model, optimizer, scheduler, train_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, test_dataloader
        )
    )

    # Register the LR scheduler for checkpointing
    accelerator.register_for_checkpointing(scheduler)

    # Initial accuracy evaluation before training begins to establish a baseline.
    train_acc = eval(
        model, train_dataloader, device
    )  # Evaluate model on the training set.
    test_acc = eval(model, test_dataloader, device)  # Evaluate model on the test set.
    # Record the initial (pre-training) accuracies and epoch number.
    data["epoch"].append(0)
    data["test_acc"].append(test_acc)
    data["train_acc"].append(train_acc)

    # Initialize the progress bar for visual feedback during training.
    progress_bar = tqdm(total=epochs, desc="Training", position=0, leave=True)
    for epoch in range(1, epochs + 1):  # Loop over the dataset multiple times
        # Train the model for one epoch and return the training accuracy.
        train_acc = train(model, optimizer, scheduler, train_dataloader, device)

        # Every 10 epochs, evaluate and print the model's performance.
        if not (epoch % 10):
            # Every 100 epochs, print current epoch and accuracies to console.
            if not (epoch % 100):
                accelerator.print(
                    f'Epoch: {data["epoch"][-1]}, Test Acc: {data["test_acc"][-1]}, Train Acc: {data["train_acc"][-1]}'
                )
            # Evaluate model on the test set and update tracking data.
            test_acc = eval(model, test_dataloader, device)
            data["epoch"].append(epoch)
            data["test_acc"].append(test_acc)
            data["train_acc"].append(train_acc)

        # Update the progress bar.
        progress_bar.update(1)
    # Close the progress bar once training is complete.
    progress_bar.close()

    # Print the final test accuracy after all epochs are completed.
    accelerator.print(f"Final Test Acc: {test_acc}")
    # Ensure all processes have finished before proceeding.
    accelerator.wait_for_everyone()
    # Save the model and optimizer state to disk.
    accelerator.save_state(output_dir="final_model/")

    # Save training and testing accuracy data to a CSV file for analysis.
    pd.DataFrame(data).to_csv(f"final_model/info.csv")


# Entry point for script execution.
if __name__ == "__main__":
    run()
