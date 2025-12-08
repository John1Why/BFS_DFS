class DependencyGraph:
    def __init__(self):
        self.adj = {}
        self.nodes = {}

    def add_component(self, name):
        if name not in self.nodes:
            self.nodes[name] = True
            self.adj.setdefault(name, [])

    def add_dependency(self, frm, to, weight=1.0):
        self.add_component(frm)
        self.add_component(to)
        self.adj[frm].append((to, float(weight)))

    def get_nodes(self):
        return list(self.nodes.keys())

    def _compute_indegrees(self):
        indeg = {}
        for n in self.nodes:
            indeg[n] = 0
        for u in self.adj:
            for (v, w) in self.adj[u]:
                indeg[v] = indeg.get(v, 0) + 1
        return indeg

    def get_topological_order(self):
        indeg = self._compute_indegrees()
        q = [n for n, d in indeg.items() if d == 0]
        order = []
        while q:
            n = q.pop(0)
            order.append(n)
            for (nbr, w) in list(self.adj.get(n, [])):
                indeg[nbr] -= 1
                if indeg[nbr] == 0:
                    q.append(nbr)
        if len(order) == len(self.nodes):
            return order
        else:
            return None  

    def is_acyclic(self):
        return self.get_topological_order() is not None

    def generate_dot(self):
        lines = []
        lines.append('digraph Dependencies {')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        for n in self.get_nodes():
            lines.append('  "{}";'.format(n.replace('"', '\\"')))
        for u in self.adj:
            for (v, w) in self.adj[u]:
                if w == 1.0:
                    lines.append('  "{}" -> "{}";'.format(u.replace('"','\\"'), v.replace('"','\\"')))
                else:
                    lines.append('  "{}" -> "{}" [label="{}"];'.format(u.replace('"','\\"'), v.replace('"','\\"'), w))
        lines.append('}')
        return "\n".join(lines)

    def load_from_lines(self, lines):
        added = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            left = None
            right = None
            if "зависит от" in line:
                parts = line.split("зависит от", 1)
                left = parts[0].strip()
                right = parts[1].strip()
            elif "depends on" in line:
                parts = line.split("depends on", 1)
                left = parts[0].strip()
                right = parts[1].strip()
            elif ":" in line:
                parts = line.split(":", 1)
                left = parts[0].strip()
                right = parts[1].strip()
            elif "->" in line:
                parts = line.split("->", 1)
                left = parts[0].strip()
                right = parts[1].strip()
            else:
                left = line
                right = ""
            if left:
                self.add_component(left)
            if right:
                sep_line = right.replace(" и ", ",").replace(" and ", ",")
                parts = [p.strip() for p in sep_line.split(",") if p.strip()]
                for p in parts:
                    node = p
                    weight = 1.0
                    if "(w=" in p and ")" in p:
                        try:
                            start = p.index("(w=") + 3
                            end = p.index(")", start)
                            weight = float(p[start:end])
                            node = p[:p.index("(")].strip()
                        except Exception:
                            node = p
                            weight = 1.0
                    self.add_dependency(left, node, weight)
                    added.append((left, node, weight))
        return added


class DependencyAnalyzer:
    def __init__(self, graph):
        self.graph = graph
        self._dfs_cache = {}

    def find_dependencies_bfs(self, start):
        if start not in self.graph.nodes:
            return []

        levels = []
        visited = set([start]) 
        first = [nbr for (nbr, w) in self.graph.adj.get(start, [])]
        level_nodes = []
        for n in first:
            if n not in visited:
                level_nodes.append(n)
                visited.add(n)
        if level_nodes:
            levels.append(level_nodes)

        current = level_nodes
        while current:
            next_level = []
            for node in current:
                for (nbr, w) in self.graph.adj.get(node, []):
                    if nbr not in visited:
                        next_level.append(nbr)
                        visited.add(nbr)
            if next_level:
                levels.append(next_level)
            current = next_level
        return levels

    def find_dependencies_dfs(self, start):
        if start not in self.graph.nodes:
            return set()

        if start in self._dfs_cache:
            return set(self._dfs_cache[start])

        visited_global = set()

        def dfs(node, path_set):
            if node not in self.graph.adj:
                return
            for (nbr, w) in self.graph.adj.get(node, []):
                if nbr in path_set:
                    visited_global.add(nbr)  
                    continue
                if nbr not in visited_global:
                    visited_global.add(nbr)
                    path_set.add(nbr)
                    dfs(nbr, path_set)
                    path_set.remove(nbr)
                else:
                    continue

        dfs(start, set([start]))
        self._dfs_cache[start] = set(visited_global)
        return set(visited_global)

    def critical_path(self):
        topo = self.graph.get_topological_order()
        if topo is None:
            return None

        dist = {}
        parent = {}
        for n in topo:
            dist[n] = float('-inf')
            parent[n] = None
        indeg = self.graph._compute_indegrees()
        sources = [n for n, d in indeg.items() if d == 0]
        if not sources:
            for n in topo:
                dist[n] = 0.0
        else:
            for s in sources:
                dist[s] = 0.0

        for u in topo:
            if dist[u] == float('-inf'):

                pass
            for (v, w) in self.graph.adj.get(u, []):
                base = dist[u]
                if base == float('-inf'):
                    candidate = w
                else:
                    candidate = base + w
                if candidate > dist[v]:
                    dist[v] = candidate
                    parent[v] = u

        max_node = None
        max_val = float('-inf')
        for n, val in dist.items():
            if val > max_val:
                max_val = val
                max_node = n

        if max_node is None or max_val == float('-inf'):
            return (0.0, [])

        path = []
        cur = max_node
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return (max_val, path)



def demo():
    print("=== Демонстрация системы анализа зависимостей ===\n")

    lines = [
        "A зависит от B, C",
        "B зависит от D",
        "C зависит от D, E",
        "E зависит от B"
    ]

    g = DependencyGraph()
    g.load_from_lines(lines)

    print("Список узлов:", sorted(g.get_nodes()))
    print("\nРёбра (из -> в [вес]):")
    for u in sorted(g.adj.keys()):
        for (v, w) in g.adj[u]:
            print("  {} -> {}  [{}]".format(u, v, w))

    topo = g.get_topological_order()
    if topo is None:
        print("\nТопологическая сортировка: Обнаружен цикл")
    else:
        print("\nТопологическая сортировка (порядок сборки):", topo)


    analyzer = DependencyAnalyzer(g)


    bfs_levels = analyzer.find_dependencies_bfs("A")
    print("\nBFS уровни зависимостей для 'A':")
    for i, lvl in enumerate(bfs_levels, start=1):
        print("  Уровень {}: {}".format(i, lvl))


    dfs_set = analyzer.find_dependencies_dfs("A")
    print("\nDFS (все уникальные зависимости) для 'A':", sorted(dfs_set))

    dot = g.generate_dot()
    print("\nDOT-описание графа (скопируйте в .dot и визуализируйте через graphviz):\n")
    print(dot)


    crit = analyzer.critical_path()
    if crit is None:
        print("\nКритический путь: Невозможно вычислить (граф содержит цикл)")
    else:
        length, path = crit
        print("\nКритический путь (макс. суммарный вес):")
        print("  Длина =", length)
        print("  Путь =", path)


    print("\n--- Пример с весами ---")
    g2 = DependencyGraph()

    lines2 = [
        "A: B (w=3), C (w=2)",
        "B: D (w=4)",
        "C: D (w=1), E (w=2)",
        "E: B (w=1)"
    ]
    g2.load_from_lines(lines2)
    print("Рёбра (g2):")
    for u in sorted(g2.adj.keys()):
        for (v, w) in g2.adj[u]:
            print("  {} -> {}  [{}]".format(u, v, w))
    analyzer2 = DependencyAnalyzer(g2)
    topo2 = g2.get_topological_order()
    print("\nТопологический порядок g2:", topo2)
    crit2 = analyzer2.critical_path()
    if crit2 is None:
        print("Критический путь g2: граф содержит цикл")
    else:
        print("Критический путь g2: длина = {}, путь = {}".format(crit2[0], crit2[1]))

if __name__ == "__main__":
    demo()
