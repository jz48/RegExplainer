import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
import heapq
import os
from utils.dataset_util.data_utils import preprocess_adj, adj_to_edge_index


class PrintGraph(Graph):
    """
    Example subclass of the Graph class.

    Prints activity log to file or standard output.
    """

    def __init__(self, data=None, name="", file=None, **attr):
        super().__init__(data=data, name=name, **attr)
        if file is None:
            import sys
            self.fh = sys.stdout
        else:
            self.fh = open(file, "w")

    def add_node(self, n, attr_dict=None, **attr):
        super().add_node(n, attr_dict=attr_dict, **attr)
        # self.fh.write(f"Add node: {n}\n")

    def add_nodes_from(self, nodes, **attr):
        for n in nodes:
            self.add_node(n, **attr)

    def remove_node(self, n):
        super().remove_node(n)
        # self.fh.write(f"Remove node: {n}\n")

    def remove_nodes_from(self, nodes):
        for n in nodes:
            self.remove_node(n)

    def add_edge(self, u, v, attr_dict=None, **attr):
        super().add_edge(u, v, attr_dict=attr_dict, **attr)
        # self.fh.write(f"Add edge: {u}-{v}\n")

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        for e in ebunch:
            u, v = e[0:2]
            self.add_edge(u, v, attr_dict=attr_dict, **attr)

    def remove_edge(self, u, v):
        super().remove_edge(u, v)
        # self.fh.write(f"Remove edge: {u}-{v}\n")

    def remove_edges_from(self, ebunch):
        for e in ebunch:
            u, v = e[0:2]
            self.remove_edge(u, v)

    def clear(self):
        super().clear()
        # self.fh.write("Clear graph\n")


def plot_graph(edge_index, labels):
    for i in range(len(edge_index)):
        edges = edge_index[i]
        edges = edges.tolist()
        label = labels[i]

        G = PrintGraph()

        G.add_nodes_from(range(25), weight=1)
        G.add_edges_from(zip(edges[0], edges[1]))

        pos = nx.spring_layout(G, seed=225)  # Seed for reproducible layout
        nx.draw(G, pos, node_size=5, font_size=8)
        plt.text(-0.5, 0.5, 'label: ' + str(label))
        plt.show()

        a = 1
    pass


def plot_expl(edge_index, feature, label, ground_truth, expl, save_path=None, file_name=None, if_save=False, if_show=True):
    edges = edge_index
    edges = edges
    label_dict = {}
    for i in range(len(feature)):
        label_dict[i] = feature[i][0]

    G1 = PrintGraph()
    G1.add_nodes_from(range(25), weight=1)
    G1.add_edges_from(zip(edges[0], edges[1]))

    def split_edges(edges, expls):
        edge_list1 = []
        expl_1 = []

        edge_list2 = []
        expl_2 = []

        for i in range(len(expls)):
            if edges[0][i] < edges[1][i]:
                edge_list1.append([edges[0][i], edges[1][i]])
                expl_1.append(expls[i])
            elif edges[0][i] > edges[1][i]:
                edge_list2.append([edges[0][i], edges[1][i]])
                expl_2.append(expls[i])
            else:
                continue
                assert 0
        return edge_list1, expl_1, edge_list2, expl_2

    edge_list1, expl_1, edge_list2, expl_2 = split_edges(edge_index, expl)

    expl_colors_1 = [[i, i, i] for i in expl_1]
    expl_colors_2 = [[i, i, i] for i in expl_2]

    G2 = PrintGraph()
    G2.add_nodes_from(range(25), weight=1)
    G2.add_edges_from(edge_list1)
    edge_labels_G2 = {(edge_list1[i][0], edge_list1[i][1]): "{:.5f}".format(expl_1[i]) for i in range(len(expl_1))}

    G3 = PrintGraph()
    G3.add_nodes_from(range(25), weight=1)
    G3.add_edges_from(edge_list2)
    edge_labels_G3 = {(edge_list2[i][0], edge_list2[i][1]): "{:.5f}".format(expl_2[i]) for i in range(len(expl_2))}

    def build_ground_truth_expl_color(expl_):
        color = []
        top_expl = heapq.nlargest(5, expl_)
        for i in expl_:
            if i in top_expl:
                color.append([1.0, 0.0, 0.0])
            else:
                color.append([0.0, 0.0, 1.0])
        return color

    G4 = PrintGraph()
    G4.add_nodes_from(range(25), weight=1)
    G4.add_edges_from(edge_list1)
    color4 = build_ground_truth_expl_color(expl_1)

    G5 = PrintGraph()
    G5.add_nodes_from(range(25), weight=1)
    G5.add_edges_from(edge_list2)
    color5 = build_ground_truth_expl_color(expl_2)

    plt.figure(0)
    pos1 = nx.spring_layout(G1, seed=225)  # Seed for reproducible layout
    plt.subplot(221)
    nx.draw(G1, pos1, labels=label_dict, with_labels=True, node_size=5, font_size=8)

    pos4 = nx.spring_layout(G4, seed=225)  # Seed for reproducible layout
    plt.subplot(223)
    nx.draw(G4, pos4, edge_color=color4, with_labels=True, node_size=10, font_size=3, width=5)

    pos5 = nx.spring_layout(G5, seed=225)  # Seed for reproducible layout
    plt.subplot(224)
    nx.draw(G5, pos5, edge_color=color5, with_labels=True, node_size=10, font_size=3, width=5)
    plt.title('label: ' + str(label / 10))
    if if_show:
        plt.show()
    if if_save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, file_name), dpi=500)
    plt.close()

    plt.figure(1)
    pos2 = nx.spring_layout(G2, seed=225)  # Seed for reproducible layout
    nx.draw(G2, pos2, edge_color=expl_colors_1, with_labels=True, node_size=10, font_size=3, width=5)
    nx.draw_networkx_edge_labels(
        G2, pos2,
        edge_labels=edge_labels_G2,
        font_color='red',
        font_size=5
    )
    if if_save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, '1_'+file_name), dpi=500)
    plt.close()

    plt.figure(2)
    pos3 = nx.spring_layout(G3, seed=225)  # Seed for reproducible layout
    nx.draw(G3, pos3, edge_color=expl_colors_2, with_labels=True, node_size=10, font_size=3, width=5)
    nx.draw_networkx_edge_labels(
        G3, pos3,
        edge_labels=edge_labels_G3,
        font_color='red',
        font_size=5
    )
    if if_save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, '2_'+file_name), dpi=500)
    plt.close()


def test_plot_color_edge():
    G = nx.Graph()
    G.add_edge(0, 1, color='r', weight=2)
    G.add_edge(1, 2, color='g', weight=4)
    G.add_edge(2, 3, color='b', weight=6)
    G.add_edge(3, 4, color='y', weight=3)
    G.add_edge(4, 0, color='m', weight=1)

    colors = nx.get_edge_attributes(G, 'color').values()
    weights = nx.get_edge_attributes(G, 'weight').values()

    pos = nx.circular_layout(G)
    nx.draw(G, pos,
            edge_color=colors,
            width=list(weights),
            with_labels=True,
            node_color='lightgreen')


def test():
    G = PrintGraph()
    G.add_node("foo")
    G.add_nodes_from("bar", weight=8)
    G.remove_node("b")
    G.remove_nodes_from("ar")
    print("Nodes in G: ", G.nodes(data=True))
    G.add_edge(0, 1, weight=10)
    print("Edges in G: ", G.edges(data=True))
    G.remove_edge(0, 1)
    G.add_edges_from(zip(range(0, 3), range(1, 4)), weight=10)
    print("Edges in G: ", G.edges(data=True))
    G.remove_edges_from(zip(range(0, 3), range(1, 4)))
    print("Edges in G: ", G.edges(data=True))

    G = PrintGraph()
    nx.add_path(G, range(10))
    nx.add_star(G, range(9, 13))
    pos = nx.spring_layout(G, seed=225)  # Seed for reproducible layout
    nx.draw(G, pos)
    plt.show()


if __name__ == '__main__':
    test()

