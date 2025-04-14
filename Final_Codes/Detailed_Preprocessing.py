from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

def train_model(model, train_loader, val_loader, vocab_size, epochs=5, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    actual_criterion = FocalLoss(alpha=0.75, gamma=2.0)
    suspected_criterion = FocalLoss(alpha=0.75, gamma=2.0)

    for epoch in range(epochs):
        model.train()
        train_actual_loss, train_suspected_loss = 0.0, 0.0
        train_actual_preds, train_actual_true = [], []
        train_suspected_preds, train_suspected_true = [], []

        # Use tqdm to show progress for the training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", unit="batch"):
            tokens = batch['tokens'].to(device)
            move_features = batch['move_features'].to(device)
            actual_labels = batch['labels']['actual_lie'].to(device)
            suspected_labels = batch['labels']['suspected_lie'].to(device)

            optimizer.zero_grad()
            actual_logits, suspected_logits = model(tokens, move_features, tokens != 0)

            actual_loss = actual_criterion(actual_logits, actual_labels)

            mask = suspected_labels != -1
            suspected_loss = suspected_criterion(
                suspected_logits[mask], suspected_labels[mask]
            ) if mask.sum() > 0 else torch.tensor(0.0, device=device)

            loss = actual_loss + suspected_loss
            loss.backward()
            optimizer.step()

            train_actual_loss += actual_loss.item()
            train_suspected_loss += suspected_loss.item()

            actual_preds = torch.argmax(actual_logits, dim=1).cpu().numpy()
            suspected_preds = torch.argmax(suspected_logits, dim=1).cpu().numpy()
            train_actual_preds.extend(actual_preds)
            train_actual_true.extend(actual_labels.cpu().numpy())
            train_suspected_preds.extend(suspected_preds[mask.cpu()])
            train_suspected_true.extend(suspected_labels[mask.cpu()].cpu().numpy())  # 🔧 fixed line

        # Validation with tqdm for progress bar
        model.eval()
        val_actual_macro, val_actual_lie = [], []
        val_suspected_macro, val_suspected_lie = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation", unit="batch"):
                tokens = batch['tokens'].to(device)
                move_features = batch['move_features'].to(device)
                actual_labels = batch['labels']['actual_lie'].to(device)
                suspected_labels = batch['labels']['suspected_lie'].to(device)

                actual_logits, suspected_logits = model(tokens, move_features, tokens != 0)
                actual_preds = torch.argmax(actual_logits, dim=1).cpu().numpy()
                suspected_preds = torch.argmax(suspected_logits, dim=1).cpu().numpy()

                mask = suspected_labels != -1
                val_actual_macro.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='macro'))
                val_actual_lie.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='binary', pos_label=1))

                if mask.sum() > 0:
                    val_suspected_macro.append(f1_score(
                        suspected_labels[mask.cpu()].cpu().numpy(),
                        suspected_preds[mask.cpu()],
                        average='macro'
                    ))
                    val_suspected_lie.append(f1_score(
                        suspected_labels[mask.cpu()].cpu().numpy(),
                        suspected_preds[mask.cpu()],
                        average='binary', pos_label=1
                    ))

        print(f"Epoch {epoch+1}:")
        print(f"  Train Actual Loss: {train_actual_loss/len(train_loader):.4f}, "
              f"Suspected Loss: {train_suspected_loss/len(train_loader):.4f}")
        print(f"  Train Actual Macro F1: {f1_score(train_actual_true, train_actual_preds, average='macro'):.4f}, "
              f"Lie F1: {f1_score(train_actual_true, train_actual_preds, average='binary', pos_label=1):.4f}")
        print(f"  Train Suspected Macro F1: {f1_score(train_suspected_true, train_suspected_preds, average='macro'):.4f}, "
              f"Lie F1: {f1_score(train_suspected_true, train_suspected_preds, average='binary', pos_label=1):.4f}")
        print(f"  Val Actual Macro F1: {np.mean(val_actual_macro):.4f}, "
              f"Lie F1: {np.mean(val_actual_lie):.4f}")
        print(f"  Val Suspected Macro F1: {np.mean(val_suspected_macro):.4f}, "
              f"Lie F1: {np.mean(val_suspected_lie):.4f}")


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    test_actual_macro, test_actual_lie = [], []
    test_suspected_macro, test_suspected_lie = [], []

    with torch.no_grad():
        # Use tqdm to show progress for the evaluation loop
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            tokens = batch['tokens'].to(device)
            move_features = batch['move_features'].to(device)
            actual_labels = batch['labels']['actual_lie'].to(device)
            suspected_labels = batch['labels']['suspected_lie'].to(device)

            actual_logits, suspected_logits = model(tokens, move_features, tokens != 0)
            actual_preds = torch.argmax(actual_logits, dim=1).cpu().numpy()
            suspected_preds = torch.argmax(suspected_logits, dim=1).cpu().numpy()

            mask = suspected_labels != -1
            test_actual_macro.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='macro'))
            test_actual_lie.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='binary', pos_label=1))

            if mask.sum() > 0:
                test_suspected_macro.append(f1_score(
                    suspected_labels[mask.cpu()].cpu().numpy(),
                    suspected_preds[mask.cpu()],
                    average='macro'
                ))
                test_suspected_lie.append(f1_score(
                    suspected_labels[mask.cpu()].cpu().numpy(),
                    suspected_preds[mask.cpu()],
                    average='binary', pos_label=1
                ))

    print(f"Test Actual Macro F1: {np.mean(test_actual_macro):.4f}, Lie F1: {np.mean(test_actual_lie):.4f}")
    print(f"Test Suspected Macro F1: {np.mean(test_suspected_macro):.4f}, Lie F1: {np.mean(test_suspected_lie):.4f}")

    return (
        np.mean(test_actual_macro),
        np.mean(test_actual_lie),
        np.mean(test_suspected_macro),
        np.mean(test_suspected_lie)
    )

# Usage (assuming your model, loaders, vocab etc. are defined)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightweightTransformer(vocab_size=len(vocab))  # make sure this class is defined
train_model(model, train_loader, val_loader, vocab_size=len(vocab), epochs=5, device=device)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
evaluate_model(model, test_loader, device=device)
rom sklearn.metrics import f1_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

# Assuming FocalLoss is defined elsewhere (not shown in original code)
# Example placeholder (uncomment and adjust if needed):
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
"""

def train_model(model, train_loader, val_loader, vocab_size, epochs=5, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    actual_criterion = FocalLoss(alpha=0.75, gamma=2.0)
    suspected_criterion = FocalLoss(alpha=0.75, gamma=2.0)

    for epoch in range(epochs):
        model.train()
        train_actual_loss, train_suspected_loss = 0.0, 0.0
        train_actual_preds, train_actual_true = [], []
        train_suspected_preds, train_suspected_true = [], []

        # Use tqdm to show progress for the training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", unit="batch"):
            tokens = batch['tokens'].to(device)
            move_features = batch['move_features'].to(device)
            actual_labels = batch['labels']['actual_lie'].to(device)
            suspected_labels = batch['labels']['suspected_lie'].to(device)

            optimizer.zero_grad()
            actual_logits, suspected_logits = model(tokens, move_features, tokens != 0)

            actual_loss = actual_criterion(actual_logits, actual_labels)

            mask = suspected_labels != -1
            suspected_loss = suspected_criterion(
                suspected_logits[mask], suspected_labels[mask]
            ) if mask.sum() > 0 else torch.tensor(0.0, device=device)

            loss = actual_loss + suspected_loss
            loss.backward()
            optimizer.step()

            train_actual_loss += actual_loss.item()
            train_suspected_loss += suspected_loss.item()

            actual_preds = torch.argmax(actual_logits, dim=1).cpu().numpy()
            suspected_preds = torch.argmax(suspected_logits, dim=1).cpu().numpy()
            train_actual_preds.extend(actual_preds)
            train_actual_true.extend(actual_labels.cpu().numpy())
            train_suspected_preds.extend(suspected_preds[mask.cpu()])
            train_suspected_true.extend(suspected_labels[mask.cpu()].cpu().numpy())

        # Debugging: Print unique labels for training aggregates
        print(f"Epoch {epoch+1} - Train Suspected True Labels:", np.unique(train_suspected_true))
        print(f"Epoch {epoch+1} - Train Suspected Pred Labels:", np.unique(train_suspected_preds))

        # Validation with tqdm for progress bar
        model.eval()
        val_actual_macro, val_actual_lie = [], []
        val_suspected_macro, val_suspected_lie = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation", unit="batch"):
                tokens = batch['tokens'].to(device)
                move_features = batch['move_features'].to(device)
                actual_labels = batch['labels']['actual_lie'].to(device)
                suspected_labels = batch['labels']['suspected_lie'].to(device)

                actual_logits, suspected_logits = model(tokens, move_features, tokens != 0)
                actual_preds = torch.argmax(actual_logits, dim=1).cpu().numpy()
                suspected_preds = torch.argmax(suspected_logits, dim=1).cpu().numpy()

                mask = suspected_labels != -1

                # Debugging: Print unique labels safely
                print("Validation batch - Actual labels:", np.unique(actual_labels.cpu().numpy()))
                print("Validation batch - Actual predictions:", np.unique(actual_preds))
                if mask.sum() > 0:
                    masked_suspected_labels = suspected_labels[mask].cpu().numpy()
                    masked_suspected_preds = suspected_preds[mask.cpu().numpy()]
                    print("Validation batch - Suspected labels:", np.unique(masked_suspected_labels))
                    print("Validation batch - Suspected predictions:", np.unique(masked_suspected_preds))
                else:
                    print("Validation batch - Suspected labels: [] (empty mask)")
                    print("Validation batch - Suspected predictions: [] (empty mask)")

                val_actual_macro.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='macro'))
                val_actual_lie.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='binary', pos_label=1))

                if mask.sum() > 0:
                    val_suspected_macro.append(f1_score(
                        suspected_labels[mask].cpu().numpy(),
                        suspected_preds[mask.cpu().numpy()],
                        average='macro'
                    ))
                    val_suspected_lie.append(f1_score(
                        suspected_labels[mask].cpu().numpy(),
                        suspected_preds[mask.cpu().numpy()],
                        average='binary', pos_label=1
                    ))

        print(f"Epoch {epoch+1}:")
        print(f"  Train Actual Loss: {train_actual_loss/len(train_loader):.4f}, "
              f"Suspected Loss: {train_suspected_loss/len(train_loader):.4f}")
        print(f"  Train Actual Macro F1: {f1_score(train_actual_true, train_actual_preds, average='macro'):.4f}, "
              f"Lie F1: {f1_score(train_actual_true, train_actual_preds, average='binary', pos_label=1):.4f}")
        print(f"  Train Suspected Macro F1: {f1_score(train_suspected_true, train_suspected_preds, average='macro'):.4f}")
        print(f"  Val Actual Macro F1: {np.mean(val_actual_macro):.4f}, "
              f"Lie F1: {np.mean(val_actual_lie):.4f}")
        print(f"  Val Suspected Macro F1: {np.mean(val_suspected_macro):.4f}, "
              f"Lie F1: {np.mean(val_suspected_lie):.4f}")


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    test_actual_macro, test_actual_lie = [], []
    test_suspected_macro, test_suspected_lie = [], []

    with torch.no_grad():
        # Use tqdm to show progress for the evaluation loop
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            tokens = batch['tokens'].to(device)
            move_features = batch['move_features'].to(device)
            actual_labels = batch['labels']['actual_lie'].to(device)
            suspected_labels = batch['labels']['suspected_lie'].to(device)

            actual_logits, suspected_logits = model(tokens, move_features, tokens != 0)
            actual_preds = torch.argmax(actual_logits, dim=1).cpu().numpy()
            suspected_preds = torch.argmax(suspected_logits, dim=1).cpu().numpy()

            mask = suspected_labels != -1

            # Debugging: Print unique labels safely
            print("Test batch - Actual labels:", np.unique(actual_labels.cpu().numpy()))
            print("Test batch - Actual predictions:", np.unique(actual_preds))
            if mask.sum() > 0:
                masked_suspected_labels = suspected_labels[mask].cpu().numpy()
                masked_suspected_preds = suspected_preds[mask.cpu().numpy()]
                print("Test batch - Suspected labels:", np.unique(masked_suspected_labels))
                print("Test batch - Suspected predictions:", np.unique(masked_suspected_preds))
            else:
                print("Test batch - Suspected labels: [] (empty mask)")
                print("Test batch - Suspected predictions: [] (empty mask)")

            test_actual_macro.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='macro'))
            test_actual_lie.append(f1_score(actual_labels.cpu().numpy(), actual_preds, average='binary', pos_label=1))

            if mask.sum() > 0:
                test_suspected_macro.append(f1_score(
                    suspected_labels[mask].cpu().numpy(),
                    suspected_preds[mask.cpu().numpy()],
                    average='macro'
                ))
                test_suspected_lie.append(f1_score(
                    suspected_labels[mask].cpu().numpy(),
                    suspected_preds[mask.cpu().numpy()],
                    average='binary', pos_label=1
                ))

    print(f"Test Actual Macro F1: {np.mean(test_actual_macro):.4f}, Lie F1: {np.mean(test_actual_lie):.4f}")
    print(f"Test Suspected Macro F1: {np.mean(test_suspected_macro):.4f}, Lie F1: {np.mean(test_suspected_lie):.4f}")

    return (
        np.mean(test_actual_macro),
        np.mean(test_actual_lie),
        np.mean(test_suspected_macro),
        np.mean(test_suspected_lie)
    )

# Usage (assuming your model, loaders, vocab etc. are defined)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Placeholder for LightweightTransformer (you need to define this class)
# Example placeholder (uncomment and adjust if needed):
"""
class LightweightTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes=2):  # Binary classification
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4), num_layers=2
        )
        self.fc = nn.Linear(128, num_classes)
        self.move_fc = nn.Linear(10, 128)  # Example for move_features

    def forward(self, tokens, move_features, attention_mask):
        x = self.embedding(tokens)
        move_emb = self.move_fc(move_features)
        x = x + move_emb  # Simplistic combination
        x = self.transformer(x, src_key_padding_mask=~attention_mask)
        x = x.mean(dim=1)  # Global average pooling
        actual_logits = self.fc(x)
        suspected_logits = self.fc(x)  # Same head for simplicity
        return actual_logits, suspected_logits
"""

model = LightweightTransformer(vocab_size=len(vocab))  # Ensure vocab is defined
train_model(model, train_loader, val_loader, vocab_size=len(vocab), epochs=5, device=device)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
evaluate_model(model, test_loader, device=device)
