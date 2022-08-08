def network_dos(g):
    return []


if __name__ == '__main__':
    from load_data import load_dataset

    dataset = 'aids700nef'
    align_metric = 'mcs'
    node_ordering = 'bfs'

    dataset = load_dataset(dataset, 'all', align_metric, node_ordering)
    dataset.print_stats()
    gs = dataset.gs
    print(len(gs))
    g1 = gs[0]
    g2 = gs[1]
    dos_g1 = network_dos(g1)
    dos_g2 = network_dos(g2)
    print(dos_g1)
    print(dos_g2)
