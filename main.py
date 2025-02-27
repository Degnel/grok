import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer

# =============================================================================
# Fonctions utilitaires
# =============================================================================

def get_flat_weights(model: nn.Module) -> torch.Tensor:
    """
    Retourne un vecteur 1D contenant tous les paramètres du modèle.
    """
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def generate_dataset(model: nn.Module, num_samples: int, seq_len: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Génère un dataset à partir du modèle enseignant.
    - Les entrées X sont des tokens aléatoires dans [0, vocab_size).
    - On passe X dans le modèle pour obtenir les logits (shape: [batch, vocab_size, seq_len]).
    - Les labels sont obtenus en prenant le token le plus probable (argmax sur la dimension vocabulaire).
    """
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    with torch.no_grad():
        logits = model(X)  # logits de shape (batch, vocab_size, seq_len)
    labels = logits.argmax(dim=1)  # labels de shape (batch, seq_len)
    return X, labels

from torch.utils.data import DataLoader, TensorDataset

def train_model(model: nn.Module, train_X: torch.Tensor, train_y: torch.Tensor,
                test_X: torch.Tensor, test_y: torch.Tensor, criterion,
                optimizer, max_epochs: int, batch_size: int = 16, early_stop: bool = False,
                threshold: float = 1e-3) -> tuple[nn.Module, dict]:
    """
    Entraîne le modèle avec un DataLoader en mini-batch.
    """
    history = {'train_loss': [], 'test_loss': []}
    
    # Création du DataLoader pour le dataset d'entraînement
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Moyenne de la perte sur l'ensemble des mini-batches
        history['train_loss'].append(epoch_loss / len(train_loader))

        # Évaluation sur le test set
        model.eval()
        with torch.no_grad():
            test_logits = model(test_X)
            test_loss = criterion(test_logits, test_y)
        history['test_loss'].append(test_loss.item())

        if early_stop and test_loss.item() < threshold:
            break
    
    return model, history


def analyze_weights(runs: list[tuple[torch.Tensor, torch.Tensor, dict]]):
    """
    Calcule et affiche la moyenne et la variance des poids initiaux et finaux 
    à partir d'une liste de runs.
    Chaque run est un tuple (poids_initiaux, poids_finaux, historique).
    """
    if not runs:
        print("Aucun run à analyser.")
        return
    init_weights_list = [r[0] for r in runs]
    final_weights_list = [r[1] for r in runs]
    init_stack = torch.stack(init_weights_list)
    final_stack = torch.stack(final_weights_list)
    init_mean = init_stack.mean().item()
    init_var = init_stack.var().item()
    final_mean = final_stack.mean().item()
    final_var = final_stack.var().item()
    print("Statistiques des poids:")
    print(f"  Initial: moyenne = {init_mean:.6f}, variance = {init_var:.6f}")
    print(f"  Final:   moyenne = {final_mean:.6f}, variance = {final_var:.6f}")

def analyze_distances(final_weights_list: list[torch.Tensor]):
    """
    Calcule et affiche la moyenne et l'écart-type des distances par paires entre
    les vecteurs de poids finaux sans utiliser de boucle explicite.

    Pour cela, on utilise torch.cdist qui calcule la distance euclidienne entre
    chaque paire de vecteurs. On extrait ensuite la partie supérieure de la
    matrice (hors diagonale) pour obtenir les distances uniques.
    """
    n = len(final_weights_list)
    if n < 2:
        print("Pas assez de runs pour calculer les distances.")
        return
    # Empilement des vecteurs de poids finaux (shape: [n, d])
    final_stack = torch.stack(final_weights_list)
    # Calcul de la matrice des distances (shape: [n, n])
    dist_matrix = torch.cdist(final_stack, final_stack)
    # Extraction de la partie supérieure de la matrice sans la diagonale
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    distances = dist_matrix[mask]
    mean_distance = distances.mean().item()
    std_distance = distances.std().item()
    print(f"Distances par paires entre les poids finaux: moyenne = {mean_distance:.6f}, écart-type = {std_distance:.6f}")

# =============================================================================
# Bloc principal
# =============================================================================

def main():
    torch.manual_seed(0)
    
    # Paramètres généraux
    vocab_size = 100
    seq_len = 5
    
    # Profondeurs des modèles
    teacher_depth = 2
    student_depth = 5
    
    # Paramètres calculés à partir des ratios
    # Pour le modèle enseignant
    teacher_d_model = 16 # 128 * teacher_depth
    teacher_n_heads = 2 # teacher_depth
    teacher_d_ff = 16 # 512 * teacher_depth
    
    # Pour le modèle étudiant
    student_d_model = 16 # 128 * student_depth
    student_n_heads = 2 # student_depth
    student_d_ff = 16 # 512 * student_depth
    
    # Nombre de runs et paramètres d'entraînement
    num_runs_classical = 100
    num_runs_grokking = 3
    classical_max_epochs = 20
    grokking_max_epochs = 20000
    grokking_threshold = 1e-3
    
    # Création du modèle enseignant (Transformer, petit, profondeur 2)
    teacher = Transformer(
        d_model=teacher_d_model,
        n_heads=teacher_n_heads,
        d_ff=teacher_d_ff,
        depth=teacher_depth,
        dropout=0.1,
        vocab_size=vocab_size,
        max_context_size=seq_len
    )
    
    # Création d'un modèle étudiant temporaire (Transformer, grand, profondeur 5) pour déterminer le nombre de paramètres
    temp_student = Transformer(
        d_model=student_d_model,
        n_heads=student_n_heads,
        d_ff=student_d_ff,
        depth=student_depth,
        dropout=0.1,
        vocab_size=vocab_size,
        max_context_size=seq_len
    )
    num_params_student = sum(p.numel() for p in temp_student.parameters())
    dataset_size = 50 * num_params_student
    print(f"Nombre de paramètres du modèle étudiant: {num_params_student}")
    print(f"Taille du dataset: {dataset_size}")
    
    # Génération des datasets de train et de test à partir du modèle enseignant
    train_X, train_y = generate_dataset(teacher, dataset_size, seq_len, vocab_size)
    test_X, test_y = generate_dataset(teacher, dataset_size, seq_len, vocab_size)
    
    # Critère : entropie croisée
    criterion = nn.CrossEntropyLoss()
    
    # -----------------------------------------------------------------------------
    # Entraînement classique : 100 runs sans early stopping (classical_max_epochs = 20)
    # -----------------------------------------------------------------------------
    classical_runs = []
    print("\n=== Entraînement classique ===")
    for i in range(num_runs_classical):
        student = Transformer(
            d_model=student_d_model,
            n_heads=student_n_heads,
            d_ff=student_d_ff,
            depth=student_depth,
            dropout=0.1,
            vocab_size=vocab_size,
            max_context_size=seq_len
        )
        init_weights = get_flat_weights(student).clone()
        optimizer = optim.Adam(student.parameters(), lr=1e-3)
        student, history = train_model(student, train_X, train_y, test_X, test_y,
                               criterion, optimizer, classical_max_epochs,
                               batch_size=16, early_stop=False)
        final_weights = get_flat_weights(student)
        classical_runs.append((init_weights, final_weights, history))
        print(f"Run {i+1}/{num_runs_classical} terminée, test loss final: {history['test_loss'][-1]:.6f}")
    
    # -----------------------------------------------------------------------------
    # Entraînement grokking : 3 runs avec early stopping et grokking_max_epochs = 20000
    # -----------------------------------------------------------------------------
    grokking_runs = []
    print("\n=== Entraînement grokking ===")
    for i in range(num_runs_grokking):
        student = Transformer(
            d_model=student_d_model,
            n_heads=student_n_heads,
            d_ff=student_d_ff,
            depth=student_depth,
            dropout=0.1,
            vocab_size=vocab_size,
            max_context_size=seq_len
        )
        init_weights = get_flat_weights(student).clone()
        optimizer = optim.Adam(student.parameters(), lr=1e-3)
        student, history = train_model(student, train_X, train_y, test_X, test_y,
                               criterion, optimizer, classical_max_epochs,
                               batch_size=16, early_stop=False)
        final_weights = get_flat_weights(student)
        grokking_runs.append((init_weights, final_weights, history))
        print(f"Grokking Run {i+1}/{num_runs_grokking} terminée, test loss final: {history['test_loss'][-1]:.6f}")
    
    # -----------------------------------------------------------------------------
    # Analyse des résultats
    # -----------------------------------------------------------------------------
    print("\n=== Analyse des résultats classiques ===")
    analyze_weights(classical_runs)
    classical_final_weights = [r[1] for r in classical_runs]
    analyze_distances(classical_final_weights)
    
    print("\n=== Analyse des résultats grokking ===")
    analyze_weights(grokking_runs)
    grokking_final_weights = [r[1] for r in grokking_runs]
    analyze_distances(grokking_final_weights)

if __name__ == '__main__':
    main()