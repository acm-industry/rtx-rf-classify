"""
CNN for RadioML dataset with hybrid multi-channel input.
Pipeline: [I, Q, log_power_spectrum] -> 1D CNN (BN, pooling, GAP).
Run from Classification/ with: python cnn_fft.py
"""
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def hybrid_preprocess(X_train, X_test, window='hann', log_normalize='global'):
    """
    Build 3-channel hybrid input: [I, Q, log_power_spectrum].
    X_train, X_test: (N, 2, 128). Returns (N, 3, 128) float32 each.
    log_normalize: 'global' = use train mean/std for both (recommended); None to skip.
    """
    def _log_power(X, win):
        X = np.asarray(X, dtype=np.float64)
        N, _, L = X.shape
        iq = X[:, 0, :] + 1j * X[:, 1, :]
        iq = iq * win
        spec = np.fft.fft(iq, axis=-1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        return np.log1p(power)

    L = X_train.shape[-1]
    win = np.hanning(L) if window == 'hann' else (np.hamming(L) if window == 'hamming' else np.ones(L))

    log_power_train = _log_power(X_train, win)
    log_power_test = _log_power(X_test, win)

    if log_normalize == 'global':
        mean, std = log_power_train.mean(), log_power_train.std()
        if std > 0:
            log_power_train = (log_power_train - mean) / std
            log_power_test = (log_power_test - mean) / std

    # Channels: I, Q, log_power
    I_train = np.asarray(X_train[:, 0, :], dtype=np.float32)
    Q_train = np.asarray(X_train[:, 1, :], dtype=np.float32)
    I_test = np.asarray(X_test[:, 0, :], dtype=np.float32)
    Q_test = np.asarray(X_test[:, 1, :], dtype=np.float32)

    X_hybrid_train = np.stack([I_train, Q_train, log_power_train], axis=1)   # (N, 3, 128)
    X_hybrid_test = np.stack([I_test, Q_test, log_power_test], axis=1)
    return X_hybrid_train, X_hybrid_test


class RadioMLCNN1D(nn.Module):
    """1D CNN for multi-channel input (B, in_channels, 128) with BN, pooling and GAP."""

    def __init__(self, in_channels=3, num_classes=11, dropout_rate=0.3, input_length=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 128 -> 64

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 64 -> 32

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, 128, 1)
            nn.Flatten(),             # (B, 128)
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description='RadioML CNN with hybrid I/Q + log-power input')
    parser.add_argument('--data', default='RML2016.10a_dict.pkl', help='Path to RadioML pkl')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=2016, help='Random seed')
    parser.add_argument('--save-model', default='radioml_cnn_hybrid.pth', help='Model save path')
    parser.add_argument('--save-fig', default='training_results_hybrid.png', help='Plot save path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset
    with open(args.data, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], p.keys())))), [1, 0])

    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(p[(mod, snr)])
            for _ in range(p[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    X = np.vstack(X)

    # Train/test split (stratified by (mod, snr) for balance and determinism)
    np.random.seed(args.seed)
    train_idx = []
    test_idx = []
    # lbl holds (mod, snr) aligned with X
    from collections import defaultdict
    bucket_indices = defaultdict(list)
    for i, (m, s) in enumerate(lbl):
        bucket_indices[(m, s)].append(i)
    for (m, s), idxs in bucket_indices.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        n_bucket = len(idxs)
        n_bucket_train = int(n_bucket * 0.7)
        train_idx.extend(idxs[:n_bucket_train].tolist())
        test_idx.extend(idxs[n_bucket_train:].tolist())

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = to_onehot([mods.index(lbl[x][0]) for x in train_idx])
    Y_test = to_onehot([mods.index(lbl[x][0]) for x in test_idx])
    Y_train_indices = np.argmax(Y_train, axis=1)
    Y_test_indices = np.argmax(Y_test, axis=1)

    # Hybrid 3-channel: [I, Q, log_power_spectrum] -> (N, 3, 128)
    X_hybrid_train, X_hybrid_test = hybrid_preprocess(
        X_train, X_test, window='hann', log_normalize='global'
    )
    X_train_torch = torch.FloatTensor(X_hybrid_train)
    X_test_torch = torch.FloatTensor(X_hybrid_test)
    Y_train_torch = torch.LongTensor(Y_train_indices)
    Y_test_torch = torch.LongTensor(Y_test_indices)

    # Log first training example for sanity-checking
    print("\nFirst training example (hybrid 3-channel, first 10):")
    print("  ch0 I[0, :10]:", X_hybrid_train[0, 0, :10])
    print("  ch1 Q[0, :10]:", X_hybrid_train[0, 1, :10])
    print("  ch2 log_power[0, :10]:", X_hybrid_train[0, 2, :10])

    print(f'\nHybrid train shape: {X_train_torch.shape}, test shape: {X_test_torch.shape}')
    print(f'Classes: {mods}')

    train_dataset = TensorDataset(X_train_torch, Y_train_torch)
    test_dataset = TensorDataset(X_test_torch, Y_test_torch)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = RadioMLCNN1D(in_channels=3, num_classes=len(mods), dropout_rate=args.dropout, input_length=128).to(device)
    print(model)
    print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters())}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = -1.0

    print('\nStarting training...')
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_model)

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(args.save_fig, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nBest Test Accuracy: {best_acc:.2f}%')
    print(f'Best model saved to {args.save_model}')


if __name__ == '__main__':
    main()
