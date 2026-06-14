import mgp


@mgp.write_proc
def api(ctx: mgp.ProcCtx) -> mgp.Record(info=str):
    parts = []
    parts.append("graph:" + ",".join(d for d in dir(ctx.graph) if not d.startswith("_")))
    v0 = None
    for x in ctx.graph.vertices:
        v0 = x
        break
    if v0 is not None:
        parts.append("vertex:" + ",".join(d for d in dir(v0) if not d.startswith("_")))
        parts.append("props:" + ",".join(d for d in dir(v0.properties) if not d.startswith("_")))
        e0 = None
        for ed in v0.out_edges:
            e0 = ed
            break
        if e0 is None:
            for ed in v0.in_edges:
                e0 = ed
                break
        if e0 is not None:
            parts.append("edge:" + ",".join(d for d in dir(e0) if not d.startswith("_")))
            parts.append("edge.type=" + str(e0.type))
            parts.append("first_v props t=" + str(v0.properties.get("t")) +
                         " phase=" + str(v0.properties.get("phase")))
    # confirm mutation API works: create a Z node, set props, then delete it
    try:
        nv = ctx.graph.create_vertex()
        nv.properties.set("t", 1)
        nv.properties.set("phase", 0.5)
        parts.append("create_vertex OK id=" + str(nv.id))
        ctx.graph.delete_vertex(nv)
        parts.append("delete_vertex OK")
    except Exception as ex:
        parts.append("mutate FAIL: " + str(ex)[:80])
    return mgp.Record(info=" || ".join(parts))
