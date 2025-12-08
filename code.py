
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# dependency_analysis.py
# Реализация анализа зависимостей (единственный файл, без внешних импортов).

class DependencyGraph:
    def __init__(self):
        # Список смежности: node -> list of (neighbor, weight)
        self.adj = {}
        # Множество всех вершин (удобно, если вершина без исходящих)
        self.nodes = {}

    def add_component(self, name):
        if name not in self.nodes:
            self.nodes[name] = True
            self.adj.setdefault(name, [])

    def add_dependency(self, frm, to, weight=1.0):
        # Ориентированное ребро frm -> to (frm зависит от to)
        self.add_component(frm)
        self.add_component(to)
        # Добавляем ребро (не предотвращаем дубликаты намеренно — можно)
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
        # Алгоритм Кана
        indeg = self._compute_indegrees()
        # очередь вершин с indeg == 0
        q = [n for n, d in indeg.items() if d == 0]
        order = []
        # simple FIFO
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
            return None  # есть цикл

    def is_acyclic(self):
        return self.get_topological_order() is not None

    def generate_dot(self):
        # Генерация DOT-строки для graphviz
        lines = []
        lines.append('digraph Dependencies {')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        # узлы
        for n in self.get_nodes():
            lines.append('  "{}";'.format(n.replace('"', '\\"')))
        # ребра с весами (если weight != 1 выводим)
        for u in self.adj:
            for (v, w) in self.adj[u]:
                if w == 1.0:
                    lines.append('  "{}" -> "{}";'.format(u.replace('"','\\"'), v.replace('"','\\"')))
                else:
                    lines.append('  "{}" -> "{}" [label="{}"];'.format(u.replace('"','\\"'), v.replace('"','\\"'), w))
        lines.append('}')
        return "\n".join(lines)

    def load_from_lines(self, lines):
        # простой парсер формата:
        # "A зависит от B, C"
        # "A depends on B, C"
        # "A: B, C"
        # "A -> B" или "A -> B, C"
        # возвращает список добавленных (from, to) для проверки
        added = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            # Разделим на левую и правую части
            # Проверим разные разделители
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
                # нераспознанная строка — возможно просто "A" (вершина без зависимостей)
                left = line
                right = ""
            if left:
                self.add_component(left)
            if right:
                # заменим 'and' и русские 'и' на запятые, уберём 'and' слова
                sep_line = right.replace(" и ", ",").replace(" and ", ",")
                # разделим по запятым
                parts = [p.strip() for p in sep_line.split(",") if p.strip()]
                for p in parts:
                    # возможны веса через "(w=...)" или "[w=..]" — простая проверка
                    node = p
                    weight = 1.0
                    # попробовать найти "(w=" и ")"
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
        # cache для DFS: node -> set(dependencies)
        self._dfs_cache = {}

    def find_dependencies_bfs(self, start):
        # Возвращает список уровней: level 1 = прямые зависимости (distance=1), level 2 = distance=2, ...
        if start not in self.graph.nodes:
            return []

        levels = []
        visited = set([start])  # чтобы не помещать старт в результаты и избежать циклов
        # начальный уровень — прямые зависимости
        first = [nbr for (nbr, w) in self.graph.adj.get(start, [])]
        # Оставим только те, которых ещё не посещали (иначе start может быть в списке)
        level_nodes = []
        for n in first:
            if n not in visited:
                level_nodes.append(n)
                visited.add(n)
        if level_nodes:
            levels.append(level_nodes)

        # BFS по уровням
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
        # Возвращает множество всех уникальных зависимостей reachable from start (исключая start).
        if start not in self.graph.nodes:
            return set()

        # Если есть в кэше, вернуть копию
        if start in self._dfs_cache:
            return set(self._dfs_cache[start])

        visited_global = set()

        def dfs(node, path_set):
            # path_set используется для предотвращения бесконечной рекурсии в цикле (локальная)
            if node not in self.graph.adj:
                return
            for (nbr, w) in self.graph.adj.get(node, []):
                if nbr in path_set:
                    # обнаружили цикл по пути, пропускаем дальнейшее углубление
                    visited_global.add(nbr)  # всё равно учитываем как зависимость
                    continue
                if nbr not in visited_global:
                    visited_global.add(nbr)
                    path_set.add(nbr)
                    dfs(nbr, path_set)
                    path_set.remove(nbr)
                else:
                    # если уже посещён глобально, всё равно можно использовать кэш
                    continue

        dfs(start, set([start]))
        # Кэшируем результат
        self._dfs_cache[start] = set(visited_global)
        return set(visited_global)

    def critical_path(self):
        # Находит самый длинный путь в DAG (по сумме весов ребер).
        # Возвращает (max_length, path_list) или None если граф содержит цикл.
        topo = self.graph.get_topological_order()
        if topo is None:
            return None

        # Инициализация расстояний
        dist = {}
        parent = {}
        for n in topo:
            dist[n] = float('-inf')
            parent[n] = None
        # источники (нулевой вход) — можно назначить 0; найдём их
        indeg = self.graph._compute_indegrees()
        sources = [n for n, d in indeg.items() if d == 0]
        if not sources:
            # Если нет явных источников, позволим любой узел стартовать с 0
            for n in topo:
                dist[n] = 0.0
        else:
            for s in sources:
                dist[s] = 0.0

        # DP по топологическому порядку
        for u in topo:
            if dist[u] == float('-inf'):
                # неприсоединённая компонента — оставляем -inf (не достижима от источников)
                # но всё равно проходим по рёбрам, чтобы распространять значения (если хотим считать локальные пути)
                pass
            for (v, w) in self.graph.adj.get(u, []):
                base = dist[u]
                if base == float('-inf'):
                    # если u ещё не инициализирован, пусть будем считать путь из u длиной w (т.е. стартуем в u)
                    candidate = w
                else:
                    candidate = base + w
                if candidate > dist[v]:
                    dist[v] = candidate
                    parent[v] = u

        # Найдём максимальное расстояние
        max_node = None
        max_val = float('-inf')
        for n, val in dist.items():
            if val > max_val:
                max_val = val
                max_node = n

        if max_node is None or max_val == float('-inf'):
            return (0.0, [])

        # восстановление пути
        path = []
        cur = max_node
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return (max_val, path)


# ============================
# Пример использования / демонстрация
# ============================
def demo():
    print("=== Демонстрация системы анализа зависимостей ===\n")

    # Пример из задания:
    # A зависит от B, C
    # B зависит от D
    # C зависит от D, E
    # E зависит от B
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

    # Топологическая сортировка
    topo = g.get_topological_order()
    if topo is None:
        print("\nТопологическая сортировка: Обнаружен цикл")
    else:
        print("\nТопологическая сортировка (порядок сборки):", topo)

    # Анализ зависимостей
    analyzer = DependencyAnalyzer(g)

    # BFS уровни для A
    bfs_levels = analyzer.find_dependencies_bfs("A")
    print("\nBFS уровни зависимостей для 'A':")
    for i, lvl in enumerate(bfs_levels, start=1):
        print("  Уровень {}: {}".format(i, lvl))

    # DFS (все уникальные зависимости) для A
    dfs_set = analyzer.find_dependencies_dfs("A")
    print("\nDFS (все уникальные зависимости) для 'A':", sorted(dfs_set))

    # Генерация DOT (для визуализации через graphviz)
    dot = g.generate_dot()
    print("\nDOT-описание графа (скопируйте в .dot и визуализируйте через graphviz):\n")
    print(dot)

    # Критический путь (longest path) в DAG
    crit = analyzer.critical_path()
    if crit is None:
        print("\nКритический путь: Невозможно вычислить (граф содержит цикл)")
    else:
        length, path = crit
        print("\nКритический путь (макс. суммарный вес):")
        print("  Длина =", length)
        print("  Путь =", path)

    # Дополнительно: пример с весами (время сборки)
    print("\n--- Пример с весами ---")
    g2 = DependencyGraph()
    # Зададим веса, например длительность сборки (A зависит от B (w=3), C (w=2))
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
