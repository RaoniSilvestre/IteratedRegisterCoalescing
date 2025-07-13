from collections import defaultdict
import time
from typing import List, Tuple, Set, Optional

from writer import load_liveness_from_json


class RegisterAllocator:
    def __init__(self, usage_sites: List[List[int]], moves: List[Tuple[int, int]], K: int):
        self.K = K
        self.usage_sites = usage_sites
        self.num_vars = len(usage_sites)

        self.initial_moves: Set[frozenset] = {frozenset(m) for m in moves}

        self.live_ranges: List[Optional[Tuple[int, int]]] = self._calculate_live_ranges() # O(n * m) where n is num of vars and m is number of usages for each  

        self.adjList: List[Set[int]] = self._build_interference_graph() # O(n²) where n is num_vars

        self.simplifyWorklist: Set[int] = set()
        self.freezeWorklist: Set[int] = set()
        self.spillWorklist: Set[int] = set()
        self.spilledNodes: Set[int] = set()
        self.coalescedNodes: Set[int] = set()
        self.coloredNodes: Set[int] = set()
        self.selectStack: List[int] = []

        self.worklistMoves: Set[frozenset] = self.initial_moves.copy()
        self.activeMoves: Set[frozenset] = set()
        self.coalescedMoves: Set[frozenset] = set()
        self.constrainedMoves: Set[frozenset] = set()
        self.frozenMoves: Set[frozenset] = set()

        self.degree = [len(neighbors) for neighbors in self.adjList]
        self.moveList = defaultdict(set)
        self.alias = {}
        self.color = {}

        self._populate_initial_worklists()

    def run(self) -> None:
        """Executes the main simplify-coalesce-freeze loop."""
        while self.simplifyWorklist or self.worklistMoves or self.freezeWorklist or self.spillWorklist:
            if self.simplifyWorklist:
                self._simplify()  
            elif self.worklistMoves:
                self._coalesce() 
            elif self.freezeWorklist:
                self._freeze()  
            elif self.spillWorklist:
                self._select_spill()
        
        self._assign_colors()

    def _simplify(self):
        n = self.simplifyWorklist.pop() # SImplify
        self.selectStack.append(n) # SelectStack
        for m in self._get_adjacent(n):
            self._decrement_degree(m) # graus

    def _coalesce(self):
        move = self.worklistMoves.pop()
        x, y = tuple(move)
        x, y = self._get_alias(x), self._get_alias(y)

        u, v = (x, y)

        if u == v:
            self.coalescedMoves.add(move)
            self._add_to_worklist(u)
        elif self._test_interference(u, v):
            self.constrainedMoves.add(move)
            self._add_to_worklist(u)
            self._add_to_worklist(v)
        elif self._conservative_heuristic(u, v):
            self.coalescedMoves.add(move)
            self._combine(u, v)
            self._add_to_worklist(u)
        else:
            self.activeMoves.add(move)

    def _freeze(self):
        u = self.freezeWorklist.pop() # Freeze.remove
        self.simplifyWorklist.add(u) # Add simplify
        self._freeze_moves(u)

    def _select_spill(self):
        # Heuristic: spill node with highest degree. 
        # TODO: Rewrite to get the spill cost (range / usages), in this case, would be
        # usage_sites[n].last() - usage_sites[n].first() / len(usage_sites)
        # (Usage sites is ordered)
        node_to_spill = max(self.spillWorklist, key=lambda n: self.degree[n])
        
        self.spillWorklist.remove(node_to_spill)
        self.simplifyWorklist.add(node_to_spill)
        self._freeze_moves(node_to_spill)

    def _assign_colors(self):
        while self.selectStack:
            n = self.selectStack.pop()
            ok_colors = set(range(self.K))
            for w in self.adjList[n]:
                aliased_w = self._get_alias(w)
                if aliased_w in self.coloredNodes: 
                    if self.color.get(aliased_w) in ok_colors:
                        ok_colors.remove(self.color[aliased_w])
            
            if not ok_colors:
                self.spilledNodes.add(n)
            else:
                self.coloredNodes.add(n)
                self.color[n] = min(ok_colors) 
        
        for n in self.coalescedNodes:
            self.color[n] = self.color[self._get_alias(n)]

    def _get_adjacent(self, n: int) -> Set[int]:
        return self.adjList[n] - set(self.selectStack) - self.coalescedNodes

    def _get_node_moves(self, n: int) -> Set[frozenset]:
        return self.moveList[n] & (self.activeMoves | self.worklistMoves)

    def _is_move_related(self, n: int) -> bool:
        return bool(self._get_node_moves(n))

    def _decrement_degree(self, m: int):
        d = self.degree[m]
        self.degree[m] -= 1
        if d == self.K:
            self._enable_moves({m} | self._get_adjacent(m))
            if m in self.spillWorklist:
                self.spillWorklist.remove(m)
            if self._is_move_related(m):
                self.freezeWorklist.add(m)
            else:
                self.simplifyWorklist.add(m)

    def _enable_moves(self, nodes: Set[int]):
        for n in nodes:
            for move in self._get_node_moves(n):
                if move in self.activeMoves:
                    self.activeMoves.remove(move)
                    self.worklistMoves.add(move)

    def _combine(self, u: int, v: int):
        if v in self.freezeWorklist: 
            self.freezeWorklist.remove(v)
        else: 
            self.spillWorklist.remove(v)
        
        self.coalescedNodes.add(v)
        self.alias[v] = u
        self.moveList[u].update(self.moveList[v])
        
        for t in self._get_adjacent(v):
            self._add_edge(t, u)
            self._decrement_degree(t)
        
        if self.degree[u] >= self.K and u in self.freezeWorklist:
            self.freezeWorklist.remove(u)
            self.spillWorklist.add(u)

    def _add_edge(self, u: int, v: int):
        if not self._test_interference(u, v) and u != v:
            self.adjList[u].add(v)
            self.adjList[v].add(u)
            self.degree[u] += 1
            self.degree[v] += 1

    def _freeze_moves(self, u: int):
        for move in self._get_node_moves(u):
            x, y = tuple(move)
            v = y if u == x else x
            
            if move in self.activeMoves: self.activeMoves.remove(move)
            else: self.worklistMoves.remove(move)
            self.frozenMoves.add(move)

            if not self._is_move_related(v) and self.degree[v] < self.K:
                if v in self.freezeWorklist: self.freezeWorklist.remove(v)
                self.simplifyWorklist.add(v)

    def _add_to_worklist(self, u: int):
        if not self._is_move_related(u) and self.degree[u] < self.K:
            if u in self.freezeWorklist: self.freezeWorklist.remove(u)
            self.simplifyWorklist.add(u)

    def _get_alias(self, n: int) -> int:
        if n in self.coalescedNodes:
            self.alias[n] = self._get_alias(self.alias[n])
            return self.alias[n]
        return n
    
    def _test_interference(self, u: int, v: int) -> bool:
        return v in self.adjList[u]

    def _conservative_heuristic(self, u: int, v: int) -> bool:
        significant_neighbors = 0
        for neighbor in self._get_adjacent(u) | self._get_adjacent(v):
            if self.degree[neighbor] >= self.K:
                significant_neighbors += 1
        return significant_neighbors < self.K

    def _calculate_live_ranges(self):
        ranges = []
        for sites in self.usage_sites:
            if not sites: ranges.append(None)
            else: ranges.append((min(sites), max(sites)))
        return ranges

    def _build_interference_graph(self):
        adj = [set() for _ in range(self.num_vars)]
        for i in range(self.num_vars):
            for j in range(i + 1, self.num_vars):
                if frozenset({i, j}) in self.initial_moves: continue
                range_i, range_j = self.live_ranges[i], self.live_ranges[j]
                if not range_i or not range_j: continue
                start_i, end_i = range_i
                start_j, end_j = range_j
                if start_i <= end_j and start_j <= end_i:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def _populate_initial_worklists(self):
        for move in self.worklistMoves:
            u, v = tuple(move)
            self.moveList[u].add(move)
            self.moveList[v].add(move)
        for i in range(self.num_vars):
            if self.degree[i] >= self.K: self.spillWorklist.add(i)
            elif self.moveList[i] & self.worklistMoves: self.freezeWorklist.add(i)
            else: self.simplifyWorklist.add(i)




def rewrite_program(
    original_usages: List[List[int]],
    original_moves: List[Tuple[int, int]],
    spilled_nodes: Set[int]
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    print(f"\nRewriting program. Spilling nodes: {spilled_nodes.__sizeof__()}")
    new_usages = []
    new_moves = []
    
    index_map = {}
    next_new_index = 0

    for i, sites in enumerate(original_usages):
        if i not in spilled_nodes:
            index_map[i] = next_new_index
            new_usages.append(sites)
            next_new_index += 1
        else:
            for site in sites:
                new_usages.append([site])
                next_new_index += 1
    
    for u, v in original_moves:
        if u not in spilled_nodes and v not in spilled_nodes:
            new_moves.append((index_map[u], index_map[v]))
            
    return new_usages, new_moves


def run_complete_allocation(usage_sites: List[List[int]], moves: List[Tuple[int, int]], K: int):
    iteration = 1
    current_usages = usage_sites
    current_moves = moves


    print("-" * 45)
    print(f"Total variables in first iteration: {len(current_usages)}")
    print(f"Total moves in first iteration: {len(current_moves)}")
    print("-" * 45)

    while True:
        print(f"--- Allocation Iteration {iteration} ---")
        
        allocator = RegisterAllocator(current_usages, current_moves, K)
        allocator.run()

        print(f"Iteration {iteration} done.")

        if not allocator.spilledNodes:
            print("\nColoring successful with no spills.")
            print(f"Final Coloring (new indices): {allocator.color.__sizeof__()}")
            break
        else:
            if(iteration == 4):
                print(f"Final Coloring (new indices): {allocator.color.__sizeof__()}")
                print(f"Spilled nodes at the end: {allocator.spilledNodes.__sizeof__()}")
                break

            current_usages, current_moves = rewrite_program(
                current_usages,
                current_moves,
                allocator.spilledNodes
            )
            iteration += 1


            def f(it):
                return len(it) != 1

            not_splited = len(list(filter(f, current_usages)))
            splitted = len(current_usages) - not_splited

            print(f"Current usages splited: {splitted} ~ Current moves: {len(current_moves)}")
            print(f"Current usages not splited: {not_splited} ~ Current moves: {len(current_moves)}")
            print(f"Total variables in next iteration: {len(current_usages)}")
            print("-" * 45)

if __name__ == '__main__':
    liveness1 = load_liveness_from_json("data/liveness-1.json")
    liveness2 = load_liveness_from_json("data/liveness-2.json")
    liveness3 = load_liveness_from_json("data/liveness-3.json")
    liveness4 = load_liveness_from_json("data/liveness-4.json")

    liveness = [liveness1, liveness2, liveness3, liveness4]

    K = 32

    for i, item in enumerate(liveness):
        print("="*45)
        T1 = time.perf_counter()
        run_complete_allocation(usage_sites=item["usage_sites"], moves=item["moves"], K=32)
        T2 = time.perf_counter()

        elapsed_time = T2 - T1

        print("-"*45)
        print(f"--- Time {i+1}: {elapsed_time} ---")
        print("-"*45)

