@jit(type="fragment")
def shader(f):
    i,j= builtins.FragCoord.xy
    return f(i,j)

drawer = context.create_drawer()
color = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "gray", "black"]
for k in range(10):

    @jit(type="device")
    def weave(i,j):
        # k is captured
        return i + j + k % len(color)

    @jit(type="device")
    def img(i, j):
        return color[weave(i, j)]
    drawer.update(shader(img))
    drawer.show()
