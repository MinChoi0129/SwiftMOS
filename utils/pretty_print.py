def shprint(*obj):
    n = len(obj)
    print("---------------------------------" * n)
    for o in obj:
        try:
            print(o.shape, end=" | ")
        except:
            print(o, end=" | ")
    print()
    print("---------------------------------" * n)
