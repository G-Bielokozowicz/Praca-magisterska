import pandas as pd

# Wczytaj plik CSV
file_path = 'saved_results/results_graph_level.csv'
df = pd.read_csv(file_path, delimiter=';')

# Posortuj DataFrame według kolumny 'test_accuracy'
df_sorted = df.sort_values(by='test_accuracy', ascending=False)

# Zapisz posortowany DataFrame do pliku CSV
output_file_path = 'saved_results/sorted_results_graph_level.csv'
df_sorted.to_csv(output_file_path, sep=';', index=False)

print(f'Posortowany plik został zapisany w: {output_file_path}')
