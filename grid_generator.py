import csv
import numpy as np

def writeCSV(filename, list):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(list)

defaultGrid = [
    [3, 0, 0, 0, 5, 0, 0, 0, 2],
    [0, 0, 1, 0, 0, 0, 9, 0, 0],
    [6, 2, 0, 7, 0, 8, 0, 4, 1],
    [0, 0, 7, 3, 0, 0, 8, 0, 0],
    [1, 0, 0, 0, 4, 0, 0, 0, 3],
    [0, 0, 3, 0, 0, 5, 1, 0, 0],
    [8, 5, 0, 9, 0, 1, 0, 3, 6],
    [0, 0, 6, 0, 0, 0, 4, 0, 0],
    [2, 0, 0, 0, 3, 0, 0, 0, 8]]

defaultSolution = [
    [3, 9, 8, 1, 5, 4, 6, 7, 2],
    [4, 7, 1, 2, 6, 3, 9, 8, 5],
    [6, 2, 5, 7, 9, 8, 3, 4, 1],
    [5, 6, 7, 3, 1, 9, 8, 2, 4],
    [1, 8, 2, 6, 4, 7, 5, 9, 3],
    [9, 4, 3, 8, 2, 5, 1, 6, 7],
    [8, 5, 4, 9, 7, 1, 2, 3, 6],
    [7, 3, 6, 5, 8, 2, 4, 1, 9],
    [2, 1, 9, 4, 3, 6, 7, 5, 8]]

mediumGrid = [
    [0, 0, 0, 0, 3, 6, 0, 7, 0],
    [0, 0, 6, 0, 0, 0, 1, 0, 0],
    [3, 0, 0, 0, 5, 4, 9, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 9, 2],
    [5, 9, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 7, 0, 8, 0, 0],
    [0, 0, 0, 0, 1, 3, 0, 4, 0],
    [0, 0, 0, 7, 0, 2, 3, 5, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0]]

mediumGridSolution = [
    [1, 4, 2, 9, 3, 6, 5, 7, 8],
    [9, 5, 6, 2, 8, 7, 1, 3, 4],
    [3, 7, 8, 1, 5, 4, 9, 2, 6],
    [8, 3, 7, 5, 6, 1, 4, 9, 2],
    [5, 9, 1, 4, 2, 8, 7, 6, 3],
    [2, 6, 4, 3, 7, 9, 8, 1, 5],
    [7, 2, 5, 8, 1, 3, 6, 4, 9],
    [6, 8, 9, 7, 4, 2, 3, 5, 1],
    [4, 1, 3, 6, 9, 5, 2, 8, 7]]

emptyGrid = np.astype(np.zeros((9,9)), int)

writeCSV("defaultGrid.csv", defaultGrid)
writeCSV("defaultSolution.csv", defaultSolution)
writeCSV("mediumGrid.csv", mediumGrid)
writeCSV("mediumGridSolution.csv", mediumGridSolution)
writeCSV("emptyGrid.csv", emptyGrid)