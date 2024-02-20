import argparse
import numpy as np
from sklearn.cluster import OPTICS
from scipy.spatial.distance import euclidean as d_euclidean

import pickle
import os
import warnings

# UTILITY FUNCTIONS

def load_trajectories(filepath):
    """
        Load the trajectories from a pickle file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError("File not found at {}".format(filepath))

    with open(filepath, 'rb') as f:
        trajectories = pickle.load(f)

    return trajectories

def save_results(trajectories, partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories, filepath):
    """
        Save the results to a pickle file.
    """
    results = {
        'trajectories': trajectories,
        'partitions': partitions,
        'segments': segments,
        'dist_matrix': dist_matrix,
        'clusters': clusters,
        'cluster_assignments': cluster_assignments,
        'representative_trajectories': representative_trajectories
    }
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


# ? Esta funcion no es usada mas que para preprocesado de los datos
def sub_sample_trajectory(trajectory, sample_n=30):
    """
        Sub sample a trajectory to a given number of points.
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be of type np.ndarray")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be of shape (n, 2)")

    include = np.linspace(0, trajectory.shape[0]-1, sample_n, dtype=np.int32)
    return trajectory[include]

# Usado para calcular la distancia entre puntos en una trayectoria.
# Tampoco se usa durante la ejecucion del algoritmo traclus
def calculate_line_euclidean_length(line):
    """
        Calculate the euclidean length of a all points in the line.
    """
    total_length = 0
    for i in range(0, line.shape[0]):
        if i == 0:
            continue
        total_length += d_euclidean(line[i-1], line[i])

    return total_length

# ! Funcion utilizada para calcular la distancia perpendicular y paralela entre dos lineas
def get_point_projection_on_line(point, line):
    """
        Get the projection of a point on a line.
    """

    # Get the slope of the line using the start and end points
    line_slope = (line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if line[-1, 0] != line[0, 0] else np.inf

    # In case the slope is infinite, we can directly get the projection
    if np.isinf(line_slope):
        return np.array([line[0,0], point[1]])
    
    # Convert the slope to a rotation matrix
    R = slope_to_rotation_matrix(line_slope)

    # Rotate the line and point
    rot_line = np.matmul(line, R.T)
    rot_point = np.matmul(point, R.T)

    # Get the projection
    proj = np.array([rot_point[0], rot_line[0,1]])

    # Undo the rotation for the projection
    R_inverse = np.linalg.inv(R)
    proj = np.matmul(proj, R_inverse.T)

    return proj

# ! Usada para convertir una particion a una lista de segmentos
def partition2segments(partition):
    """
        Convert a partition to a list of segments.
    """

    if not isinstance(partition, np.ndarray):
        raise TypeError("partition must be of type np.ndarray")
    elif partition.shape[1] != 2:
        raise ValueError("partition must be of shape (n, 2)")
    
    segments = []
    for i in range(partition.shape[0]-1):
        segments.append(np.array([[partition[i, 0], partition[i, 1]], [partition[i+1, 0], partition[i+1, 1]]]))

    return segments

################# EQUATIONS #################

# Euclidean Distance : Accepts two points of type np.ndarray([x,y])
# DEPRECATED IN FAVOR OF THE SCIPY IMPLEMENTATION OF THE EUCLIDEAN DISTANCE
# d_euclidean = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Perpendicular Distance
def d_perpendicular(l1, l2):
    """
        Calculate the perpendicular distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    lehmer_1 = d_euclidean(l_shorter[0], ps)
    lehmer_2 = d_euclidean(l_shorter[-1], pe)

    if lehmer_1 == 0 and lehmer_2 == 0:
        return 0
    return (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)#, ps, pe, l_shorter[0], l_shorter[-1]
    
# Parallel Distance
def d_parallel(l1, l2):
    """
        Calculate the parallel distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    parallel_1 = min(d_euclidean(l_longer[0], ps), d_euclidean(l_longer[-1], ps))
    parallel_2 = min(d_euclidean(l_longer[0], pe), d_euclidean(l_longer[-1], pe))

    return min(parallel_1, parallel_2)

# Angular Distance
def d_angular(l1, l2, directional=True):
    """
        Calculate the angular distance between two lines.
    """

    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    # Get the minimum intersecting angle between both lines
    shorter_slope = (l_shorter[-1,1] - l_shorter[0,1]) / (l_shorter[-1,0] - l_shorter[0,0]) if l_shorter[-1,0] - l_shorter[0,0] != 0 else np.inf
    longer_slope = (l_longer[-1,1] - l_longer[0,1]) / (l_longer[-1,0] - l_longer[0,0]) if l_longer[-1,0] - l_longer[0,0] != 0 else np.inf

    # The case of a vertical line
    theta = None
    if np.isinf(shorter_slope):
        # Get the angle of the longer line with the x-axis and subtract it from 90 degrees
        tan_theta0 = longer_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    elif np.isinf(longer_slope):
        # Get the angle of the shorter line with the x-axis and subtract it from 90 degrees
        tan_theta0 = shorter_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    else:
        tan_theta0 = (shorter_slope - longer_slope) / (1 + shorter_slope * longer_slope)
        tan_theta1 = tan_theta0 * -1

        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))

        theta = min(theta0, theta1)

    if directional:
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])

    if 0 <= theta < (90 * np.pi / 180):
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
    elif (90 * np.pi / 180) <= theta <= np.pi:
        return np.sin(theta)
    else:
        raise ValueError("Theta is not in the range of 0 to 180 degrees.")

# Total Trajectory Distance
def distance(l1, l2, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):
    """
        Get the total trajectory distance using all three distance formulas.
    """

    perpendicular_distance = d_perpendicular(l1, l2)
    parallel_distance = d_parallel(l1, l2)
    angular_distance = d_angular(l1, l2, directional=directional)

    return (w_perpendicular * perpendicular_distance) + (w_parallel * parallel_distance) + (w_angular * angular_distance)

# ! Minimum Description Length (MDL)
# ! Identifica puntos significativos en las trayectorias donde ocurren cambios notables en el movimiento, lo cual puede indicar un cambio de comportamiento o dirección.
def minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=1, w_perpendicular=1, par=True, directional=True):
    """
        Calculate the minimum description length.
    """
    LH = LDH = 0
    for i in range(start_idx, curr_idx-1):
        ed = d_euclidean(trajectory[i], trajectory[i+1])
        LH += max(0, np.log2(ed, where=ed>0))
        if par:
            for j in range(start_idx, i-1):
                # print()
                # print(np.array([trajectory[start_idx], trajectory[i]]))
                # print(np.array([trajectory[j], trajectory[j+1]]))
                LDH += w_perpendicular * d_perpendicular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]))
                LDH += w_angular * d_angular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]), directional=directional)
    if par:
        return LH + LDH
    return LH

# ? Esta funcion tampoco se usa durante la ejecucion del algoritmo traclus
# Slope to angle in degrees
def slope_to_angle(slope, degrees=True):
    """
        Convert slope to angle in degrees.
    """
    if not degrees:
        return np.arctan(slope)
    return np.arctan(slope) * 180 / np.pi

# Slope to rotation matrix
def slope_to_rotation_matrix(slope):
    """
        Convert slope to rotation matrix.
    """
    return np.array([[1, slope], [-slope, 1]])

# Get cluster majority line orientation
def get_average_direction_slope(line_list):
    """
        Get the cluster majority line orientation.
        Returns 1 if the lines are mostly vertical, 0 otherwise.
    """
    # Get the average slopes of all the lines
    slopes = []
    for line in line_list:
        slopes.append((line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if (line[-1, 0] - line[0, 0]) != 0 else 0)
    slopes = np.array(slopes)

    # Get the average slope
    return np.mean(slopes)

# ? Otra funcion que sirve para preprocesar los datos, no es necesaria durante la ejecucion del algoritmo traclus
# Trajectory Smoothing
def smooth_trajectory(trajectory, window_size=5):
    """
        Smooth a trajectory using a moving average filter.
    """
    # Ensure that the trajectory is a numpy array of shape (n, 2)
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be a numpy array of shape (n, 2)")

    # Ensure that the window size is an odd integer
    if not isinstance(window_size, int):
        raise TypeError("Window size must be an integer")
    elif window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer")

    # Pad the trajectory with the first and last points
    padded_trajectory = np.zeros((trajectory.shape[0] + (window_size - 1), 2))
    padded_trajectory[window_size // 2:window_size // 2 + trajectory.shape[0]] = trajectory
    padded_trajectory[:window_size // 2] = trajectory[0]
    padded_trajectory[-window_size // 2:] = trajectory[-1]

    # Apply the moving average filter
    smoothed_trajectory = np.zeros(trajectory.shape)
    for i in range(trajectory.shape[0]):
        smoothed_trajectory[i] = np.mean(padded_trajectory[i:i + window_size], axis=0)

    return smoothed_trajectory

# ! Funcion clave en el rendimento del algoritmo, se encarga de particionar las trayectorias
def get_distance_matrix(partitions, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1, progress_bar=False):
    """
    Calcula la matriz de distancias entre todas las particiones de trayectorias.
    """
    # Determina el número total de particiones para dimensionar la matriz de distancias.
    n_partitions = len(partitions)
    # Inicializa la matriz de distancias con ceros, con tamaño n_partitions x n_partitions.
    dist_matrix = np.zeros((n_partitions, n_partitions))
    
    # Itera sobre todas las combinaciones posibles de particiones para calcular las distancias.
    for i in range(n_partitions):
        if progress_bar: print(f'Progreso: {i+1}/{n_partitions}', end='\r')
        for j in range(i+1):
            # Calcula la distancia entre la partición i y la partición j usando la función de distancia definida.
            # La función de distancia considera las ponderaciones para distancias perpendiculares, paralelas y angulares,
            # así como la orientación direccional si es relevante.
            dist_matrix[i,j] = dist_matrix[j,i] = distance(partitions[i], partitions[j], directional=directional, w_perpendicular=w_perpendicular, w_parallel=w_parallel, w_angular=w_angular)
            if progress_bar: print(f'Progreso: {i+1}/{n_partitions}', end='\r')

    # Asegura que los valores en la diagonal principal sean cero, pues es la distancia de una partición a sí misma.
    for i in range(n_partitions):
        dist_matrix[i,i] = 0

    # Verifica si hay valores NaN en la matriz de distancias, lo cual puede indicar un error en el cálculo.
    if np.isnan(dist_matrix).any():
        warnings.warn("La matriz de distancias contiene valores NaN")
        # Reemplaza los valores NaN con un valor máximo para evitar errores en el proceso de agrupamiento.
        dist_matrix[np.isnan(dist_matrix)] = 9999999

    # Devuelve la matriz de distancias calculada.
    return dist_matrix

#############################################

# ! El particionamiento de las trayectorias se lleva a cabo teniendo en cuanta los puntos significativos de la trayectoria
def partition(trajectory, directional=True, progress_bar=False, w_perpendicular=1, w_angular=1):
    """
    Particiona una trayectoria en segmentos significativos, buscando los puntos donde 
    la trayectoria cambia de dirección o comportamiento de manera notable.
    """

    # Verifica que la trayectoria sea un arreglo de NumPy con la forma correcta (n, 2),
    # donde n es el número de puntos y 2 representa las coordenadas x, y de cada punto.
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("La trayectoria debe ser un arreglo de NumPy")
    elif trajectory.shape[1] != 2:
        raise ValueError("La trayectoria debe tener la forma (n, 2)")

    # Inicializa la lista de índices de puntos característicos, comenzando con el primer punto de la trayectoria.
    cp_indices = [0]

    # Longitud total de la trayectoria para controlar el avance del algoritmo.
    traj_len = trajectory.shape[0]
    start_idx = 0  # Índice inicial para comenzar la evaluación de segmentos.
    
    length = 1  # Inicia la longitud del segmento a evaluar.
    while start_idx + length < traj_len:  # Mientras no se haya recorrido toda la trayectoria.
        if progress_bar:  # Si se activa la barra de progreso, muestra el porcentaje completado.
            print(f'\r{round(((start_idx + length) / traj_len) * 100, 2)}%', end='')
        
        curr_idx = start_idx + length  # Define el índice actual hasta donde se evalúa el segmento.

        # Calcula el costo de MDL para el segmento actual con y sin considerar una partición adicional.
        cost_par = minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=w_angular, w_perpendicular=w_perpendicular, directional=directional)
        cost_nopar = minimum_desription_length(start_idx, curr_idx, trajectory, par=False, directional=directional)

        if cost_par > cost_nopar:  # Si el costo con partición es mayor que sin ella,
            cp_indices.append(curr_idx-1)  # añade el último punto del segmento a los índices de puntos característicos.
            start_idx = curr_idx-1  # Actualiza el índice inicial para la próxima evaluación.
            length = 1  # Reinicia la longitud del segmento a evaluar.
        else:
            length += 1  # Si no, incrementa la longitud del segmento a evaluar.
    
    # Añade el último punto de la trayectoria a los puntos característicos.
    cp_indices.append(len(trajectory) - 1)
    
    # Devuelve los puntos de la trayectoria correspondientes a los índices de puntos característicos.
    return np.array([trajectory[i] for i in cp_indices])


# ! Representa la trayectoria promedio de un cluster
def get_representative_trajectory(lines, min_lines=3):
    """
    Obtiene la trayectoria representativa a partir de un conjunto de líneas.
    """
    # Calcula la pendiente promedio de todas las líneas para obtener una dirección general de movimiento.
    average_slope = get_average_direction_slope(lines)
    # Convierte esta pendiente en una matriz de rotación para alinear las líneas con el eje x.
    rotation_matrix = slope_to_rotation_matrix(average_slope)

    # Rota todas las líneas para que queden paralelas al eje x.
    rotated_lines = []
    for line in lines:
        rotated_lines.append(np.matmul(line, rotation_matrix.T))

    # Recolecta todos los puntos de inicio y fin de las líneas rotadas.
    starting_and_ending_points = []
    for line in rotated_lines:
        starting_and_ending_points.append(line[0])
        starting_and_ending_points.append(line[-1])
    starting_and_ending_points = np.array(starting_and_ending_points)

    # Ordena estos puntos por su coordenada x para preparar el algoritmo de línea de barrido.
    starting_and_ending_points = starting_and_ending_points[starting_and_ending_points[:, 0].argsort()]

    # Inicia el algoritmo de línea de barrido para encontrar puntos representativos.
    representative_points = []
    for p in starting_and_ending_points:
        # Cuenta cuántas líneas contienen el valor x del punto actual 'p'.
        num_p = 0
        for line in rotated_lines:
            point_sorted_line = line[line[:, 0].argsort()]
            if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                num_p += 1

        # Si el número de líneas es igual o mayor a min_lines, calcula el valor y promedio para 'p'.
        if num_p >= min_lines:
            y_avg = 0
            for line in rotated_lines:
                point_sorted_line = line[line[:, 0].argsort()]
                if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                    y_avg += (point_sorted_line[0, 1] + point_sorted_line[-1, 1]) / 2
            y_avg /= num_p
            # Añade el punto 'p' y su valor y promedio a los puntos representativos.
            representative_points.append(np.array([p[0], y_avg]))

    # Si no se encontraron puntos representativos, retorna un arreglo vacío.
    if len(representative_points) == 0:
        warnings.warn("ADVERTENCIA: No se encontraron puntos representativos.")
        return np.array([])

    # Deshace la rotación de los puntos representativos para alinearlos con la orientación original de las líneas.
    representative_points = np.array(representative_points)
    representative_points = np.matmul(representative_points, np.linalg.inv(rotation_matrix).T)
    
    # Retorna los puntos representativos, que juntos forman la trayectoria representativa del cluster.
    return representative_points



def traclus(trajectories, max_eps=None, min_samples=10, directional=True, use_segments=True, clustering_algorithm=OPTICS, mdl_weights=[1,1,1], d_weights=[1,1,1], progress_bar=False):
    """
    Implementación del algoritmo TRACLUS para el agrupamiento de trayectorias.
    """
    # Verificar que las trayectorias sean una lista de arreglos NumPy con la forma correcta (n, 2)
    if not isinstance(trajectories, list):
        raise TypeError("Las trayectorias deben ser una lista")
    for trajectory in trajectories:
        if not isinstance(trajectory, np.ndarray):
            raise TypeError("Las trayectorias deben ser una lista de arreglos NumPy")
        elif len(trajectory.shape) != 2 or trajectory.shape[1] != 2:
            raise ValueError("Las trayectorias deben tener la forma (n, 2)")

    # Particionar las trayectorias
    if progress_bar:
        print("Particionando trayectorias...")
    partitions = []
    for i, trajectory in enumerate(trajectories):
        if progress_bar:
            print(f"\rTrayectoria {i + 1}/{len(trajectories)}", end='')
        # Particionar cada trayectoria usando la función de partición, con los pesos especificados
        partitions.append(partition(trajectory, directional=directional, progress_bar=False, w_perpendicular=mdl_weights[0], w_angular=mdl_weights[2]))
    if progress_bar:
        print()

    # Convertir las particiones en segmentos si es necesario
    # ! Deveria ser siempre necesario por como funciona TRACLUS no se en que momento no lo seria
    segments = []
    if use_segments:
        if progress_bar:
            print("Convirtiendo trayectorias particionadas en segmentos...")
        for parts in partitions:
            # Convertir cada partición en segmentos de línea
            segments += partition2segments(parts)
    else:
        segments = partitions

    # Calcular la matriz de distancias entre los segmentos, calcula todas las distancias antes de empezar a agrupar
    # ! Aqui radica gran parte de la complejidad ya que por cada segmento que añadas este tendra que tener la distancia con todos los anteriores
    if progress_bar:
        print("Calculando matriz de distancias...")
    dist_matrix = get_distance_matrix(segments, directional=directional, w_perpendicular=d_weights[0], w_parallel=d_weights[1], w_angular=d_weights[2], progress_bar=progress_bar)

    # Agrupar las particiones usando el algoritmo de agrupamiento especificado, en esta caso OPTICS
    if progress_bar:
        print("Agrupando particiones...")
    clusters = []
    clustering_model = clustering_algorithm(max_eps=max_eps, min_samples=min_samples) if max_eps is not None else clustering_algorithm(min_samples=min_samples)
    cluster_assignments = clustering_model.fit_predict(dist_matrix)
    for c in range(min(cluster_assignments), max(cluster_assignments) + 1):
        # Crear clusters basados en las asignaciones del algoritmo de agrupamiento
        clusters.append([segments[i] for i in range(len(segments)) if cluster_assignments[i] == c])

    if progress_bar:
        print()

    # Obtener las trayectorias representativas de cada cluster
    if progress_bar:
        print("Obteniendo trayectorias representativas...")
    representative_trajectories = []
    for cluster in clusters:
        # Calcular la trayectoria representativa para cada cluster
        representative_trajectories.append(get_representative_trajectory(cluster))
    if progress_bar:
        print()

    # Devolver los resultados del proceso de agrupamiento
    return partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories

# Create the script version that takes in a file path for inputs
if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Trajectory Clustering Algorithm")
    parser.add_argument("input_file", help="The input file path (pickle format)")
    parser.add_argument("output_file", help="The output file path (pickle format)")
    parser.add_argument("-e", "--eps", help="The epsilon value for the clustering algorithm", type=float, default=2)
    parser.add_argument("-m", "--min_samples", help="The minimum samples value for the clustering algorithm", type=int, default=3)
    parser.add_argument("-p", "--progress_bar", help="Show the progress bar", action="store_true")
    args = parser.parse_args()

    # Load the trajectories
    trajectories = load_trajectories(args.input_file)

    # Run the TraClus algorithm
    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = traclus(trajectories, eps=args.eps, min_samples=args.min_samples, progress_bar=args.progress_bar)

    # Save the results
    save_results(args.output_file, trajectories, partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories)