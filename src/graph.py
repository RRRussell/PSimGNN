import networkx as nx


class OurGraph(object):
    def __init__(self, nxgraph):
        # if 'gid' not in nxgraph.graph or type(nxgraph.graph['gid']) is not int \
        #         or nxgraph.graph['gid'] < 0:
        #     raise ValueError('Graph ID must be non-negative integers {}'.
        #                      format(nxgraph.graph.get('gid')))
        # if not nx.is_connected(nxgraph):
        #     raise ValueError('Graph {} must be connected'.
        #                      format(nxgraph.graph['gid']))
        self.nxgraph = nxgraph

    def type(self):
        raise NotImplementedError()

    def gid(self):
        return self.nxgraph.graph['gid']

    def get_nxgraph(self):
        return self.nxgraph

    def get_image(self):
        raise ValueError('Image data does not exist in {}'.
                         format(self.__class__.__name__))

    def get_complete_graph(self):
        raise ValueError('Complete graph data does not exist in {}'.
                         format(self.__class__.__name__))

    def get_nodes_num(self):
        return len(self.nxgraph.nodes())


class BioGraph(OurGraph):
    def __init__(self, nxgraph, connected=True, require_connected=False):
        if 'gid' not in nxgraph.graph or type(nxgraph.graph['gid']) is not int \
                or nxgraph.graph['gid'] < 0:
            raise ValueError('Graph ID must be non-negative integers {}'.
                             format(nxgraph.graph.get('gid')))
        if require_connected and not nx.is_connected(nxgraph):
            raise ValueError('Graph {} must be connected'.
                             format(nxgraph.graph['gid']))
        self.is_connected = connected
        self.nxgraph = nxgraph

    def type(self):
        return 'bio_graph'


class RegularGraph(OurGraph):
    def type(self):
        return 'regular_graph'


class HierarchicalGraph(OurGraph):
    def __init__(self, TODO):
        # TODO
        return

    def gid(self):
        return None

    def get_nxgraph(self):
        return None  # TODO: return a list of networkx graphs

    def type(self):
        return 'HierarchicalGraph'


class ImageGraph(OurGraph):
    def __init__(self, delaunay_nxgraph, complete_nxgraph, image):
        super(ImageGraph, self).__init__(delaunay_nxgraph)
        self.compete_graph = complete_nxgraph
        self.image = image

    def type(self):
        return 'image_graph'

    def get_image(self):
        return self.image

    def get_complete_graph(self):
        return self.compete_graph
