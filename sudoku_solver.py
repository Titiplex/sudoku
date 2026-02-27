from typing import Tuple, List, Dict

import numpy as np
import csv
import sys
import time
import heapq


def readCSV(filename) -> np.ndarray:
    """
    Méthode pour lire un fichier CSV
    
    :param filename: path/filename du fichier
    :return: numpy ndarray de la grille sudoku
    :rtype: ndarray[9x9, int]
    """
    grid = []
    with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            cleaned_row = [int(x) if x.strip() else 0 for x in row]
            grid.append(cleaned_row)
    return np.array(grid)


def print_grid(grid: np.ndarray) -> None:
    """
    Helper method pour afficher la grille

    :param grid: Grille à afficher
    :type grid: 9x9 np.ndarray
    """
    print("-" * 25)
    for i in range(9):
        row_str = "| "
        for j in range(9):
            val = grid[i][j] if grid[i][j] != 0 else "."
            row_str += str(val) + " "
            if (j + 1) % 3 == 0:
                row_str += "| "
        print(row_str)
        if (i + 1) % 3 == 0:
            print("-" * 25)


class SudokuSolver:
    def __init__(self):
        return

    def is_valid_move(self, grid: np.ndarray, row: int, col: int, number: int) -> bool:
        """
        Vérifie si placer le number au point (row, col) dans la grille grid est permis
        
        :param grid: grille de sudoku
        :type grid: 9x9 np.ndarray
        :param row: rangée
        :type row: int
        :param col: colonne
        :type col: int
        :param number: nombre à placer dans la grille
        :type number: int
        :return: permission de placer le nombre dans la grille
        :rtype: bool
        """

        # TODO

        sub_grid = grid[row // 3 * 3:(row // 3 * 3) + 3, col // 3 * 3:(col // 3 * 3) + 3]
        for i in range(9):
            if grid[row][i] == number or grid[i][col] == number or sub_grid[i // 3, i % 3] == number: return False

        return True

    def find_empty(self, grid: np.ndarray) -> tuple[int, int]:
        """
        Retourne la première case vide de la grille
        
        :param grid: grille de sudoku
        :type grid: 9x9 np.ndarray
        :return: position de la première case vide. None si aucune case vide
        :rtype: tuple[int, int]
        """
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return i, j
        return None

    def is_valid_grid(self, grid: np.ndarray) -> bool:
        """
        Verifie si la grille contient des erreurs
        
        :param grid: grille de sudoku
        :type grid: 9x9 np.ndarray
        :return: si la grille ne contient aucune erreur
        :rtype: bool
        
        """
        for r in range(9):
            for c in range(9):
                val = grid[r][c]
                if val == 0: continue

                grid[r][c] = 0
                if not self.is_valid_move(grid, r, c, val):
                    grid[r][c] = val
                    return False
                grid[r][c] = val
        return True

    def check_solution(self, grid: np.ndarray) -> bool:
        """
        Vérifie si la grille est complète et si elle contient des erreurs
        
        :param grid: grille de sudoku
        :type grid: 9x9 np.ndarray
        :return: si la grille ne contient aucune erreur et est remplie
        :rtype: bool
        
        """

        if self.find_empty(grid) is not None:
            print("Solution Incomplete.")
            return False

        valid_grid = self.is_valid_grid(grid)
        if valid_grid:
            print("Solution Valid!")
            return True
        return False

    def solve_DFS(self, cur_grid: np.ndarray) -> np.ndarray:
        print(f"\n--- Uninformed Search DFS---")
        start_time = time.time()
        nodes_expanded = 0

        # TODO
        # Implémenter l'algorithme de recherche profondeur d'abord ici.

        stack: List[Tuple[int, int, List[int], int]] = []
        grid = cur_grid.copy()

        while True:
            pos, candidats = self.get_best_node(grid)

            if pos is None:
                cur_grid = grid
                break

            r, c = pos

            # dead end -> back track
            if len(candidats) == 0:
                while stack:
                    pr, pc, pcands, idx = stack[-1]
                    idx += 1

                    if idx < len(pcands):
                        grid[pr][pc] = pcands[idx]
                        nodes_expanded += 1
                        stack[-1] = pr, pc, pcands, idx
                    else:
                        grid[pr][pc] = 0
                        stack.pop()
                if not stack: break
                continue

            grid[r][c] = candidats[0]
            nodes_expanded += 1
            stack.append((r, c, candidats, 0))

        print(f"Time: {time.time() - start_time:.4f}s")
        print(f"Nodes Expanded: {nodes_expanded}")

        return cur_grid

    def heuristic(self, grid: np.ndarray) -> int:
        """
        Heuristic function h(n).
        Returns the number of empty cells remaining.
        (Lower is better).
        """

        # TODO
        # Implémenter une heuristique admissible et cohérente
        return int(np.count_nonzero(grid == 0))

        # return 0  # Heuristique triviale

    def solve_AStar(self, grid: np.ndarray) -> np.ndarray:
        print(f"\n--- A* Search ---")
        start_time = time.time()
        nodes_expanded = 0

        # TODO
        # Implémenter l'algorithme A* qui utilise la fonction de cout g et l'heuristique h pour une fonction d'évaluation f=g+h qui classe les noeuds à explorer
        # Utilisez la méthode heuristic() que vous avez implémenté plus haut
        # Hint: utilisez heapq pour une liste qui garde l'ordre croissant automatiquement

        start_grid = grid.copy()
        start = tuple(map(tuple, start_grid.tolist()))

        ctr = 0
        h0 = self.heuristic(start_grid)
        heap = [(0 + h0, h0, 0, ctr, start_grid, start)]
        best = {start: 0}

        while heap:
            f, h, cost, _, current, cur_key = heapq.heappop(heap)
            nodes_expanded += 1

            if self.find_empty(current) is None and self.is_valid_grid(current):
                grid = current
                break

            pos, candidats = self.get_best_node(current)
            if pos is None: continue

            r, c = pos

            for n in candidats:
                if not self.is_valid_move(current, r, c, n): continue

                next = current.copy()
                next[r][c] = n
                next_key = tuple(map(tuple, next.tolist()))

                next_g = cost + 1
                if next_key in best and next_g >= best[next_key]: continue

                best[next_key] = next_g
                next_h = self.heuristic(next)
                next_f = next_g + next_h
                ctr += 1
                heapq.heappush(heap, (next_f, next_h, next_g, ctr, next, next_key))

        print(f"Time: {time.time() - start_time:.4f}s")
        print(f"Nodes Expanded: {nodes_expanded}")

        return grid

    def get_best_node(self, grid: np.ndarray) -> tuple[tuple[int, int], list]:

        # TODO
        # Implémenter l'optimisation pour l'algorithme vorace qui choisi la meilleure case à remplir et les nombres possibles à mettre dedans

        pos = None
        cand = None
        len_ = 10

        for r in range(9):
            for c in range(9):
                if grid[r][c] != 0: continue

                candidates = []
                for n in range(1, 10):
                    if self.is_valid_move(grid, r, c, n): candidates.append(n)

                    if len(candidates) == 0: return (r, c), []
                    if len(candidates) < len_:
                        len_ = len(candidates)
                        pos = r, c
                        cand = candidates

                        if len_ == 1: return pos, cand

        if pos is None: return self.find_empty(grid), [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return pos, cand

    def solve_greedy(self, grid: np.ndarray) -> np.ndarray:
        print(f"\n--- Greedy Best-First Search ---")
        start_time = time.time()
        nodes_expanded = 0

        # TODO
        # Implémenter l'algorithme de recherche vorace qui choisi la meilleure case à remplir (utilisez get_best_node())

        start_grid = grid.copy()
        start = tuple(map(tuple, start_grid.tolist()))

        ctr = 0
        heap = [(self.heuristic(start_grid), ctr, start_grid, start)]
        visited = {start}

        while heap:
            h, _, current, curr_key = heapq.heappop(heap)
            nodes_expanded += 1

            if self.find_empty(current) is None and self.is_valid_grid(current):
                grid = current
                break

            pos, cand = self.get_best_node(current)
            if pos is None: continue

            r, c = pos

            for n in cand:
                if not self.is_valid_move(current, r, c, n): continue
                next = current.copy()
                next[r][c] = n
                next_key = tuple(map(tuple, next.tolist()))

                if next_key in visited: continue

                visited.add(next_key)
                ctr += 1
                heapq.heappush(heap, (self.heuristic(next), ctr, next, next_key))

        print(f"Time: {time.time() - start_time:.4f}s")
        print(f"Nodes Expanded: {nodes_expanded}")
        return grid


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sudoku solver (with 3 different search algorithms)')
    parser.add_argument('--search_algorithm', default='Full_Run',
                        help='which search algorithm to use bewtween "uniformed", "A*", and "greedy" (default A*)')
    parser.add_argument('--grid', default='defaultGrid.csv', help='filename of the grid to solve')

    args = parser.parse_args()
    solver = SudokuSolver()

    try:
        grid = readCSV(args.grid)
        print_grid(grid)
    except FileNotFoundError:
        print(f'File {args.grid} was not found')
        sys.exit(1)

    sol = []
    if args.search_algorithm == "uninformed":
        sol = solver.solve_DFS(grid=grid)
    elif args.search_algorithm == "A*":
        sol = solver.solve_AStar(grid=grid)
    elif args.search_algorithm == "greedy":
        sol = solver.solve_greedy(grid=grid)
    else:  # Full run
        # DFS
        sol = solver.solve_DFS(grid)
        print_grid(sol)
        print(solver.check_solution(sol))

        # A*
        sol = solver.solve_AStar(grid)
        print_grid(sol)
        print(solver.check_solution(sol))

        # Greedy
        sol = solver.solve_greedy(grid)

    print_grid(sol)
    print(solver.check_solution(sol))
