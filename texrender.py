#!/usr/bin/env python3
# call via `blender --background --factory-startup --python thisfile.py -- -m <file>.fbx
#

import argparse
from glob import glob
import os
import re
import sys

from typing import Dict, List, Tuple

# _debug = 0


print("argv: %s" % (sys.argv[1:]))


# Sadly, execBlender can't be defined later on, so it ends up having to
# sit right in the middle of our imports!
#
# ? Should we redirect stdout/stderr before execing blender?
def execBlender(reason: str):
    blender_bin = "blender"

    mypath = os.path.realpath(__file__)

    print("Not running under blender (%s)" % (reason))
    print("Re-execing myself under blender (blender must exist in path)...")

    blender_args = [
        blender_bin,
        "--background",
        "--factory-startup",
        "--python",
        mypath,
        "--",
    ] + sys.argv[1:]

    print("executing: %s" % " ".join((blender_args)))

    # For some reason, this just doesn't work under Windows if there's a
    # space in the path. Can't manage to feed it anything that will actually
    # work, despite the same command line as I can run by hand.
    try:
        os.execvp(blender_bin, blender_args)
    except OSError as e:
        print("Couldn't exec blender: %s" % (e))
        sys.exit(1)


# Check if we're running under Blender ... and if not, fix that.
# We both have to check to make sure we can import bpy, *and* check
# to make sure there's something meaningful inside that module (like
# an actual context) because there exist 'stub' bpy modules for
# developing outside of blender, that will still import just fine...)
try:
    import bpy
except ImportError:
    execBlender("no bpy available")

# It imported ok, so now check to see if we have a context object
if bpy.context is None:
    execBlender("no context available")

# We have to do the above before trying to import other things,
# because some other things might not be installed on system
# python
from mathutils import Vector

tags = {
    "color": "diffuse diff albedo base col color".split(),
    "sss_color": "sss subsurface".split(),
    "metallic": "metallic metalness metal mtl".split(),
    "specular": "specularity specular spec spc".split(),
    "normal": "normal nor nrm nrml norm".split(),
    "bump": "bump bmp".split(),
    "rough": "roughness rough rgh".split(),
    "gloss": "gloss glossy glossiness".split(),
    "displacement": "displacement displace disp dsp height heightmap".split(),
}

# split our texture filenames up into components
def split_into__components(fname):
    # Split filename into components
    # 'WallTexture_diff_2k.002.jpg' -> ['Wall', 'Texture', 'diff', 'k']
    # Remove extension
    fname = os.path.splitext(fname)[0]
    # Remove digits
    fname = ''.join(i for i in fname if not i.isdigit())
    # Separate CamelCase by space
    fname = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", fname)
    # Replace common separators with SPACE
    seperators = ['_', '.', '-', '__', '--', '#']
    for sep in seperators:
        fname = fname.replace(sep, ' ')

    components = fname.split(' ')
    components = [c.lower() for c in components]
    return components


# Look through texture_types and set value as filename of first matched file
def match_files_to_socket_names(filenames):
    socketnames = [
        ['Displacement', tags["displacement"], None],
        ['Base Color', tags["color"], None],
        ['Subsurface Color', tags["sss_color"], None],
        ['Metallic', tags["metallic"], None],
        ['Specular', tags["specular"], None],
        ['Roughness', tags["rough"] + tags["gloss"], None],
        ['Normal', tags["normal"] + tags["bump"], None],
    ]

    for sname in socketnames:
        for fname in filenames:
            filenamecomponents = split_into__components(fname)
            matches = set(sname[1]).intersection(
                set(filenamecomponents))
            # TODO: ignore basename (if texture is named "fancy_metal_nor", it
            # will be detected as metallic map, not normal map)
            if matches:
                sname[2] = fname
                break
    return socketnames


# set thing up with material ... stolen pretty much directly from node
# wrangler, because node wrangler itself made too many assumptions about
# the state of things, and I got tired of fighting with it
def scene_prep(args, files):
    bpy.ops.wm.open_mainfile(
        filepath=args.scene, load_ui=False, use_scripts=False)

    nodes = bpy.data.materials['Preview Material'].node_tree.nodes
    links = bpy.data.materials['Preview Material'].node_tree.links
    pbsdf_shader = nodes["Principled BSDF"]

    # filenames = [
    #     r"pbrtex\red_brick_wall_4_diffuse.png",
    #     r"pbrtex\red_brick_wall_4_glossiness.png",
    #     r"pbrtex\red_brick_wall_4_height.png",
    #     r"pbrtex\red_brick_wall_4_normal.png",
    #     r"pbrtex\red_brick_wall_4_reflection.png",
    # ]

    filenames = []
    for f in files:
        if '*' in f:
            filenames += glob(f)
        else:
            filenames += f

    # pprint(filenames)

    # Remove sockets without files
    sockets = match_files_to_socket_names(filenames)

    # FIXME: don't check for file existing, for testing
    # sockets = [s for s in sockets if s[2] and os.path.exists(s[2])]
    sockets = [s for s in sockets if s[2]]

    if not sockets:
        print("ERROR: No matching images found")
        return False

    pbsdf_shader.inputs["Specular"].default_value = 0.0

    print("Matched textures:")
    texture_nodes = []
    disp_texture = None
    normal_node = None
    roughness_node = None

    for i, sname in enumerate(sockets):
        print(i, sname[0], sname[2])

        # DISPLACEMENT NODES
        if sname[0] == 'Displacement':
            disp_texture = nodes.new(type='ShaderNodeTexImage')
            # img = bpy.data.images.load(path.join(import_path, sname[2]))
            # img = bpy.data.images.load(os.path.join(os.getcwd(), sname[2]))
            img = bpy.data.images.load(os.path.realpath(sname[2]))
            disp_texture.image = img
            disp_texture.label = 'Displacement'
            if disp_texture.image:
                disp_texture.image.colorspace_settings.is_data = True

            # Add displacement offset nodes
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.inputs["Scale"].default_value = 0.15
            disp_node.location = pbsdf_shader.location + Vector((0, -710))
            link = links.new(disp_node.inputs[0], disp_texture.outputs[0])

            # could be 'DISPLACEMENT', 'BUMP' or 'BOTH'
            # FIXME: add subdivision
            bpy.data.materials['Preview Material'].cycles.displacement_method = 'BOTH'

            # Find output node
            output_node = [n for n in nodes if n.bl_idname ==
                           'ShaderNodeOutputMaterial']
            if output_node:
                if not output_node[0].inputs[2].is_linked:
                    link = links.new(
                        output_node[0].inputs[2], disp_node.outputs[0])

            continue

        if not pbsdf_shader.inputs[sname[0]].is_linked:
            # No texture node connected -> add texture node with new image
            texture_node = nodes.new(type='ShaderNodeTexImage')
            # img = bpy.data.images.load(path.join(import_path, sname[2]))
            img = bpy.data.images.load(os.path.realpath(sname[2]))

            texture_node.image = img

            # NORMAL NODES
            if sname[0] == 'Normal':
                # Test if new texture node is normal or bump map
                fname_components = split_into__components(sname[2])
                match_normal = set(tags["normal"]).intersection(
                    set(fname_components))
                match_bump = set(tags["bump"]).intersection(
                    set(fname_components))
                if match_normal:
                    # If Normal add normal node in between
                    normal_node = nodes.new(type='ShaderNodeNormalMap')
                    link = links.new(
                        normal_node.inputs[1], texture_node.outputs[0])
                elif match_bump:
                    # If Bump add bump node in between
                    normal_node = nodes.new(type='ShaderNodeBump')
                    link = links.new(
                        normal_node.inputs[2], texture_node.outputs[0])

                link = links.new(
                    pbsdf_shader.inputs[sname[0]], normal_node.outputs[0])
                normal_node_texture = texture_node

            elif sname[0] == 'Roughness':
                # Test if glossy or roughness map
                fname_components = split_into__components(sname[2])
                match_rough = set(tags["rough"]).intersection(
                    set(fname_components))
                match_gloss = set(tags["gloss"]).intersection(
                    set(fname_components))

                if match_rough:
                    # If Roughness nothing to to
                    link = links.new(
                        pbsdf_shader.inputs[sname[0]], texture_node.outputs[0])

                elif match_gloss:
                    # If Gloss Map add invert node
                    invert_node = nodes.new(type='ShaderNodeInvert')
                    link = links.new(
                        invert_node.inputs[1], texture_node.outputs[0])

                    link = links.new(
                        pbsdf_shader.inputs[sname[0]], invert_node.outputs[0])
                    roughness_node = texture_node

            else:
                # This is a simple connection Texture --> Input slot
                link = links.new(
                    pbsdf_shader.inputs[sname[0]], texture_node.outputs[0])

            # Use non-color for all but 'Base Color' Textures
            if not sname[0] in ['Base Color'] and texture_node.image:
                texture_node.image.colorspace_settings.is_data = True

        else:
            # If already texture connected. add to node list for alignment
            texture_node = pbsdf_shader.inputs[sname[0]].links[0].from_node

        # This are all connected texture nodes
        texture_nodes.append(texture_node)
        texture_node.label = sname[0]

    if disp_texture:
        texture_nodes.append(disp_texture)

    # Alignment
    for i, texture_node in enumerate(texture_nodes):
        offset = Vector((-700, (i * -280) + 200))
        texture_node.location = pbsdf_shader.location + offset

    if normal_node:
        # Extra alignment if normal node was added
        normal_node.location = normal_node_texture.location + Vector((300, 0))

    if roughness_node:
        # Alignment of invert node if glossy map
        invert_node.location = roughness_node.location + Vector((300, 0))

    # Add texture input + mapping
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = pbsdf_shader.location + Vector((-1200, 0))
    if len(texture_nodes) > 1:
        # If more than one texture add reroute node in between
        reroute = nodes.new(type='NodeReroute')
        texture_nodes.append(reroute)
        tex_coords = Vector((texture_nodes[0].location.x, sum(
            n.location.y for n in texture_nodes) / len(texture_nodes)))
        reroute.location = tex_coords + Vector((-50, -120))
        for texture_node in texture_nodes:
            link = links.new(texture_node.inputs[0], reroute.outputs[0])
        link = links.new(reroute.inputs[0], mapping.outputs[0])
    # if len(texture_nodes) > 1:
    #     for texture_node in texture_nodes:
    #         link = links.new(texture_node.inputs[0], mapping.outputs[0])
    else:
        link = links.new(texture_nodes[0].inputs[0], mapping.outputs[0])

    # Connect texture_coordiantes to mapping node
    texture_input = nodes.new(type='ShaderNodeTexCoord')
    texture_input.location = mapping.location + Vector((-200, 0))
    link = links.new(mapping.inputs[0], texture_input.outputs[2])

    # Create frame around tex coords and mapping
    frame = nodes.new(type='NodeFrame')
    frame.label = 'Mapping'
    mapping.parent = frame
    texture_input.parent = frame
    frame.update()

    # Create frame around texture nodes
    frame = nodes.new(type='NodeFrame')
    frame.label = 'Textures'
    for tnode in texture_nodes:
        tnode.parent = frame
    frame.update()

    # Just to be sure
    pbsdf_shader.select = False
    nodes.update()
    links.update()
    bpy.data.materials['Preview Material'].node_tree.update_tag()

    return True


output_re = re.compile("(.*).(png|jpg|jpeg)", re.IGNORECASE)

def render(args, outfile):
    # output_file = os.path.join(os.path.realpath("."), f"{basename}.png")
    # render.set_output_properties(scene=scene, resolution_percentage=100,
    #                              output_file_path=output_file)

    # num_samples = 16
    # render.set_cycles_renderer(scene, camera,
    #                            num_samples, use_denoising=False)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_percentage = 100
    scene.render.filepath = os.path.realpath(outfile)
    # scene.view_layers[0].cycles.use_denoising = True
    scene.cycles.samples = 128

    m = output_re.match(outfile)
    basename = m.group(1)
    filetype = str(m.group(2))

    if filetype == "png":
        scene.render.image_settings.file_format = "PNG"
    elif filetype == "jpg" or filetype == "jpeg":
        scene.render.image_settings.file_format = "JPEG"
    else:
        print(f"ERROR: unknown file extension for {os.path.basename(outfile)}")
        return

    # FIXME: Should this be in main or somewhere else that's not here?
    if args.keep_blend:
        bpy.ops.wm.save_mainfile(filepath=basename + ".blend")

    bpy.ops.render.render(
        animation=False, write_still=True, use_viewport=False)


def readable_file(f: str) -> str:
    if not os.path.isfile(f) or not os.access(f, os.R_OK):
        raise argparse.ArgumentError(f"'{f}' is not a readable file")

    return f


def main(argv):
    input_name = ""
    # print("fbxregroup version: %s" % (git_version()))

    print(argv)

    # When we get called from blender, the entire blender command line is
    # passed to us as argv. Arguments for us specifically are separated
    # with a double dash, which makes blender stop processing arguments.
    # If there's not a double dash in argv, it means we can't possibly
    # have any arguments, in which case, we should blow up.
    if (("--" in argv) == False):
        print("Usage: blender --background --python thisfile.py -- <file>.fbx")
        return 1

    # chop argv down to just our arguments
    args_start = argv.index("--") + 1
    argv = argv[args_start:]

    parser = argparse.ArgumentParser(
        # FIXME: We need to specify this, because we have no meaningful
        # argv[0], but we probably shouldn't just hardcode it
        prog='texrender.py',
        description='Render a preview of a (PBR) texture',
    )

    parser.add_argument("--debug", "-d", action="count", default=0)

    mydir = os.path.realpath(os.path.dirname(__file__))
    default_scene_file = os.path.join(mydir, "scenes", "texrender_scene.blend")

    parser.add_argument(
        "-s",
        "--scene",
        help="blender scene file to use for rendering",
        type=readable_file,
        default=default_scene_file,
    )

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="preview.png",

        help="file to output material preview to",
    )

    parser.add_argument(
        "-k",
        "--keep-blend",
        default=False,
        action='store_true',
        help="save a copy of the generated .blend file",
    )

    parser.add_argument(
        "files",
        help="specify files to process",
        metavar="file",
        type=str,  # FIXME: Is there a 'file' type arg?
        nargs="+",
    )

    args = parser.parse_args(argv)

    global _debug
    _debug = args.debug

    if not scene_prep(args, args.files):
        print("ERROR: Scene prep failed")
    else:
        render(args, args.out)

    # bpy.ops.wm.save_mainfile(filepath="test.blend")
    bpy.ops.wm.quit_blender()
    sys.exit(0)  # Shouldn't be reached


if __name__ == "__main__":
    sys.exit(main(sys.argv))