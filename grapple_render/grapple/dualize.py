"""
dualize.py -- turn a single-robot MJCF into a two-robot scene.

Renders the momentum-projected run as the solid robot and the recoil-naive
run as a translucent ghost, in the SAME scene, sharing ONE target: same
target, same start, so the only difference is the warm start.

The ghost is made collision-free (contype=0 conaffinity=0) so the two copies
never interact -- they are two independent executions drawn on top of each
other, not a physical scene with two spacecraft.

    python -m grapple.dualize in.xml out.xml --root base --prefix gh_
"""

import argparse
import copy
import xml.etree.ElementTree as ET

# every element type whose "name" lives in a MuJoCo namespace we might clone
_NAMED = ("body", "joint", "freejoint", "geom", "site", "camera", "light",
          "tendon", "sensor")
# attributes that reference a name we may have renamed
_REFS = ("joint", "joint1", "joint2", "body", "body1", "body2", "site",
         "site1", "site2", "geom", "geom1", "geom2", "objname")

GHOST_RGBA = "0.62 0.68 0.78 0.30"
GHOST_ACCENT_RGBA = "0.74 0.80 0.90 0.42"

# Decorative geometry duplicated on the ghost is pure clutter: two sets of
# solar panels tell the reader nothing and hide the arms. Drop any ghost geom
# whose name contains one of these. Structural links + gripper always survive.
GHOST_DROP = ("panel", "boom", "foil", "band")


def _collect_names(elem, names):
    if elem.tag in _NAMED and "name" in elem.attrib:
        names.add(elem.attrib["name"])
    for c in elem:
        _collect_names(c, names)


def _prune_ghost(elem, drop):
    """Remove decorative duplicates from the ghost copy, in place."""
    for parent in elem.iter():
        for c in list(parent):
            if c.tag == "geom":
                nm = c.attrib.get("name", "").lower()
                if any(k in nm for k in drop):
                    parent.remove(c)


def _rename(elem, names, prefix, ghost_style=True):
    if "name" in elem.attrib and elem.attrib["name"] in names:
        elem.attrib["name"] = prefix + elem.attrib["name"]
    for r in _REFS:
        if r in elem.attrib and elem.attrib[r] in names:
            elem.attrib[r] = prefix + elem.attrib[r]

    if ghost_style and elem.tag == "geom":
        # visual-only: never collide, never contribute contacts
        elem.attrib["contype"] = "0"
        elem.attrib["conaffinity"] = "0"
        # rgba overrides the referenced material when it differs from the
        # MuJoCo default, which is what gives us transparency
        cls = elem.attrib.get("class", "")
        elem.attrib["rgba"] = GHOST_ACCENT_RGBA if cls == "vis" else GHOST_RGBA

    for c in elem:
        _rename(c, names, prefix, ghost_style)


def dualize(in_path, out_path, root_body="base", prefix="gh_", drop_decor=True):
    tree = ET.parse(in_path)
    root = tree.getroot()

    wb = root.find("worldbody")
    if wb is None:
        raise SystemExit("no <worldbody> in model")

    src = None
    for b in wb.iter("body"):
        if b.attrib.get("name") == root_body:
            src = b
            break
    if src is None:
        raise SystemExit(f"no body named {root_body!r}; pass --root")

    names = set()
    _collect_names(src, names)

    ghost = copy.deepcopy(src)
    if drop_decor:
        _prune_ghost(ghost, GHOST_DROP)
    _rename(ghost, names, prefix, ghost_style=True)
    wb.append(ghost)

    # clone actuators that drive the cloned joints
    act = root.find("actuator")
    if act is not None:
        for a in list(act):
            tgt = a.attrib.get("joint")
            if tgt in names:
                g = copy.deepcopy(a)
                if "name" in g.attrib:
                    g.attrib["name"] = prefix + g.attrib["name"]
                g.attrib["joint"] = prefix + tgt
                act.append(g)

    # keep any explicit contact excludes consistent
    con = root.find("contact")
    if con is not None:
        for c in list(con):
            g = copy.deepcopy(c)
            touched = False
            for r in ("body1", "body2", "geom1", "geom2"):
                if g.attrib.get(r) in names:
                    g.attrib[r] = prefix + g.attrib[r]
                    touched = True
            if touched:
                con.append(g)

    tree.write(out_path, encoding="unicode")
    return out_path, sorted(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--root", default="base")
    ap.add_argument("--prefix", default="gh_")
    ap.add_argument("--keep-decor", action="store_true",
                    help="do not prune duplicated decorative geoms from the ghost")
    a = ap.parse_args()
    p, names = dualize(a.inp, a.out, a.root, a.prefix, drop_decor=not a.keep_decor)
    print(f"wrote {p}  ({len(names)} names cloned with prefix {a.prefix!r})")


if __name__ == "__main__":
    main()
