import pebbles

if __name__ == '__main__':
    meta = pebbles.moduleconfig.Metadata()
    print(meta._data_dir)
    print(meta._chains_dir)
    print(meta.products_dir)
    print(meta.cmb_thermo_fpath('tag', 8, 50))
