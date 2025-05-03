import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Data points for contour plots
list1 = [
    (1, 4, 0.127), (2, 8, 0.201), (2, 16, 0.205), (3, 11, 0.236), 
    (3, 21, 0.248), (4, 16, 0.291), (5, 13, 0.291), (6, 11, 0.287), 
    (7, 9, 0.203), (4, 32, 0.289), (5, 26, 0.305), (6, 21, 0.331), 
    (7, 18, 0.334), (8, 16, 0.336), (9, 14, 0.309), (5, 51, 0.318), 
    (6, 43, 0.342), (7, 37, 0.339), (8, 32, 0.364), (9, 28, 0.360), 
    (10, 26, 0.372), (6, 85, 0.318), (8, 64, 0.359), (10, 51, 0.385), 
    (12, 43, 0.386), (14, 37, 0.402), (16, 32, 0.411), (10, 102, 0.378),
    (13, 79, 0.401), (16, 64, 0.421), (19, 54, 0.435), (22, 47, 0.428)
]

compressed_list1 = [
    (4, 0.127), (16, 0.201), (32, 0.236), (64, 0.291), 
    (128, 0.336), (256, 0.372), (512, 0.411), (1024, 0.435)
    ]

list2 = [
    (1, 4, 0.241), (2, 8, 0.298), (2, 16, 0.309), (3, 11, 0.347), 
    (3, 21, 0.349), (4, 16, 0.387), (5, 13, 0.394), (6, 11, 0.408), 
    (7, 9, 0.401), (4, 32, 0.382), (5, 26, 0.396), (6, 21, 0.419), 
    (7, 18, 0.422), (8, 16, 0.445), (9, 14, 0.450), (5, 51, 0.401), 
    (6, 43, 0.417), (7, 37, 0.439), (8, 32, 0.440), (9, 28, 0.456), 
    (10, 26, 0.464), (6, 85, 0.423), (8, 64, 0.433), (10, 51, 0.477), 
    (12, 43, 0.489), (14, 37, 0.501), (16, 32, 0.489), (10, 102, 0.473), 
    (13, 79, 0.490), (16, 64, 0.514), (19, 54, 0.513), (22, 47, 0.526)
]

compressed_list2 = [
    (4, 0.241), (16, 0.298), (32, 0.347), (64, 0.408), 
    (128, 0.450), (256, 0.464), (512, 0.501), (1024, 0.526)
    ]

list3 = [
    (1, 4, 0.272), (2, 8, 0.316), (2, 16, 0.326), (3, 11, 0.350), 
    (3, 21, 0.362), (4, 16, 0.382), (5, 13, 0.394), (6, 11, 0.396), 
    (7, 9, 0.393), (4, 32, 0.379), (5, 26, 0.394), (6, 21, 0.404), 
    (7, 18, 0.420), (8, 16, 0.424), (9, 14, 0.432), (5, 51, 0.402), 
    (6, 43, 0.410), (7, 37, 0.416), (8, 32, 0.408), (9, 28, 0.414), 
    (10, 26, 0.426), (11, 23, 0.441), (12, 21, 0.444), (13, 19, 0.443),
    (6, 85, 0.408), (8, 64, 0.428), (10, 51, 0.445), (12, 43, 0.456), 
    (14, 37, 0.463), (16, 32, 0.465), (18, 28, 0.474), (20, 26, 0.477), 
    (10, 102, 0.446), (13, 79, 0.459), (16, 64, 0.458), (19, 54, 0.477), 
    (22, 47, 0.483), (25, 41, 0.499)
    ]

compressed_list3 = [
    (4, 0.272), (16, 0.316),  (32, 0.350), (64, 0.396), 
    (128, 0.432), (256, 0.444), (512, 0.477), (1024, 0.499)
    ]

list4 = [
    (1, 4, 0.277), (2, 8, 0.321), (2, 16, 0.313), (3, 11, 0.356), 
    (3, 21, 0.348), (4, 16, 0.389), (5, 13, 0.394), (6, 11, 0.419), 
    (7, 9, 0.411), (4, 32, 0.370), (5, 26, 0.399), (6, 21, 0.416), 
    (7, 18, 0.424), (8, 16, 0.419), (9, 14, 0.429), (5, 51, 0.386), 
    (6, 43, 0.395), (7, 37, 0.428), (8, 32, 0.420), (9, 28, 0.447), 
    (10, 26, 0.445), (11, 23, 0.441), (12, 21, 0.432), (13, 19, 0.457), 
    (6, 85, 0.407), (8, 64, 0.431), (10, 51, 0.439), (12, 43, 0.460), 
    (14, 37, 0.474), (16, 32, 0.476), (18, 28, 0.468), (20, 26, 0.491), 
    (10, 102, 0.442), (13, 79, 0.461), (16, 64, 0.456), (19, 54, 0.452), 
    (22, 47, 0.492), (25, 41, 0.486)
]

compressed_list4 = [
    (4, 0.277), (16, 0.321), (32, 0.356), (64, 0.419), 
    (128, 0.429), (256, 0.457), (512, 0.491), (1024, 0.492)
]

list5_ = [
    (1, 4, 0.275), (2, 8, 0.358), (2, 16, 0.364), (3, 11, 0.383),
    (4, 16, 0.414), (5, 13, 0.419), (6, 11, 0.442), (7, 9, 0.436),
    (7, 18, 0.454), (8, 16, 0.469), (9, 14, 0.468), (10, 13, 0.479),
    (11, 23, 0.482), (12, 21, 0.482), (13, 20, 0.495), (14, 18, 0.508),
    (14, 37, 0.507),(16, 32, 0.512), (18, 28, 0.521), (20, 26, 0.520),
    (19, 54, 0.516),(22, 47, 0.534), (25, 41, 0.526), (28, 37, 0.520)
]

compressed_list5 = [
    (4, 0.275), (16, 0.358), (32, 0.383), (64, 0.442), 
    (128, 0.479), (256, 0.508), (512, 0.521), (1024, 0.534)
    ]

# Generate contour plots
def create_contour_plot(data_points, iteration):
    branch_factors = np.array([item[0] for item in data_points])
    num_expansions = np.array([item[1] for item in data_points])
    accuracies = np.array([item[2] for item in data_points])
    
    bf_grid = np.linspace(branch_factors.min(), branch_factors.max(), 100)
    ne_grid = np.linspace(num_expansions.min(), num_expansions.max(), 100)
    bf_mesh, ne_mesh = np.meshgrid(bf_grid, ne_grid)
    
    accuracy_interpolated = griddata(points=(branch_factors, num_expansions), values=accuracies,
                                    xi=(bf_mesh, ne_mesh), method='cubic')
    
    plt.figure(figsize=(12, 8), dpi=300)
    contour = plt.contourf(bf_mesh, ne_mesh, accuracy_interpolated, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Accuracy').ax.set_ylabel('Accuracy', fontsize=10)
    
    scatter = plt.scatter(branch_factors, num_expansions, c=accuracies, s=50, alpha=0.8,
                         cmap='viridis', edgecolors='red')
    
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.3f}', (branch_factors[i], num_expansions[i]),
                    textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    
    plt.xlabel('Branch Factor', fontsize=12)
    plt.ylabel('Number of Expansions', fontsize=12)
    plt.title(f'Accuracy as a Function of Branch Factor and Number of Expansions (Iteration {iteration})', fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.savefig(f'Branch_Factor_vs_Num_Expansions_Iteration_{iteration}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate scaling plot
def create_scaling_plot():
    def process_data(data):
        tokens = [16 * x for x, _ in data]
        accuracy = [acc * 100 for _, acc in data]
        return tokens, accuracy
    
    tokens_1, accuracy_1 = process_data(compressed_list1)
    tokens_2, accuracy_2 = process_data(compressed_list2)
    tokens_3, accuracy_3 = process_data(compressed_list3)
    tokens_4, accuracy_4 = process_data(compressed_list4)
    tokens_5, accuracy_5 = process_data(compressed_list5)
    
    plt.figure(figsize=(12, 8), dpi=300)
    plt.semilogx(tokens_1, accuracy_1, 'o-', label='Iteration 1')
    plt.semilogx(tokens_2, accuracy_2, 's-', label='Iteration 2')
    plt.semilogx(tokens_3, accuracy_3, '^-', label='Iteration 3')
    plt.semilogx(tokens_4, accuracy_4, 'D-', label='Iteration 4')
    plt.semilogx(tokens_5, accuracy_5, '^-', label='Iteration 3*')

    plt.xscale('log')
    plt.xlabel('Tokens', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Tokens across iterations', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.savefig('Accuracy_vs_Tokens_all_iterations.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate all plots
    for i, data in enumerate([list1, list2, list3, list4, list5_], 1):
        create_contour_plot(data, i)
    create_scaling_plot()

