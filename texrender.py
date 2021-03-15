#!/usr/bin/env python3
# call via `blender --background --factory-startup --python thisfile.py -- --option --option <files>
#

import argparse
from glob import glob
import os
import re
import subprocess
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

    print("executing: %s" % (" ".join((blender_args))))

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

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd, cwd=None):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(
            ['git', 'describe', '--always', '--dirty'], os.path.dirname(__file__))
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "unknown"

    return GIT_REVISION


# We have to do the above before trying to import other things,
# because some other things might not be installed on system
# python
from mathutils import Vector


# split our texture filenames up into components
def split_into_components(fname):
    # Split filename into components
    # 'WallTexture_diff_2k.002.jpg' -> ['Wall', 'Texture', 'diff', 'k']
    #
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


# filename segments for different types of map
tags = {
    "color": "diffuse diff albedo base col color basecolor".split(),
    "sss_color": "sss subsurface".split(),
    "metallic": "metallic metalness metal mtl".split(),
    "specular": "specularity specular spec spc".split(),
    "normal": "normal nor nrm nrml norm".split(),
    "bump": "bump bmp".split(),
    "rough": "roughness rough rgh".split(),
    "gloss": "gloss glossy glossiness".split(),
    "displacement": "displacement displace disp dsp height heightmap".split(),
}

# Look through texture_types and set value as filename of first matched file
def match_files_to_socket_names(filenames):
    # FIXME: reflection map ignored (for PBR ... might be needed for non-PBR)
    sockets = {
        'Displacement': [tags["displacement"], None],
        'Base Color': [tags["color"], None],
        'Subsurface Color': [tags["sss_color"], None],
        'Metallic': [tags["metallic"], None],
        'Specular': [tags["specular"], None],
        'Roughness': [tags["rough"] + tags["gloss"], None],
        'Normal': [tags["normal"] + tags["bump"], None],
    }

    matchednames = []
    for _, sdata in sockets.items():
        for fname in filenames:
            filenamecomponents = split_into_components(fname)
            # print(f"components: {filenamecomponents}")
            matches = set(sdata[0]).intersection(set(filenamecomponents))
            # TODO: ignore basename (if texture is named "fancy_metal_nor", it
            # will be detected as metallic map, not normal map)
            if matches:
                matchednames.append(fname)
                sdata[1] = fname
                break

    unmatched = list(set(filenames) - set(matchednames))

    return (sockets, unmatched)


def spec_from_ior(ior: float) -> float:
    return (((ior - 1) / (ior + 1)) ** 2) / 0.08


# set thing up with material ... stolen pretty much directly from node
# wrangler, because node wrangler itself made too many assumptions about
# the state of things, and I got tired of fighting with it
def scene_prep(args, files):
    filenames = []
    for f in files:
        if '*' in f:
            filenames += glob(f)
        else:
            filenames.append(f)

    s, uf = match_files_to_socket_names(filenames)

    # Remove sockets without files
    unused = []
    sockets = {}
    for k in s.keys():
        if s[k][1] and os.path.exists(s[k][1]):
            sockets[k] = s[k]
        else:
            unused.append(k)

    if len(sockets) == 0:
        print("ERROR: No matching images found")
        return False

    # FIXME: Maybe report this after rendering, instead?
    print("\nMatched maps:")
    for sname, sdata in s.items():
        print(f"  {sname:17} {sdata[1]}")

    if len(uf) > 0:
        print("\nUnused/unmatched files:")
        for u in uf:
            print(f"  {u}")

    if args.analyze:
        return True

    bpy.ops.wm.open_mainfile(
        filepath=args.scene, load_ui=False, use_scripts=False)

    nodes = bpy.data.materials['Preview Material'].node_tree.nodes
    links = bpy.data.materials['Preview Material'].node_tree.links
    pbsdf_shader = nodes['Principled BSDF']

    # specular/specular IoR
    if "Specular" in sockets:
        if args.specular or args.specular_ior:
            # FIXME: Do we ever want to use specular maps?
            # see https://blenderartists.org/t/the-poorly-understood-specular-slider-of-the-principled-shader/1150941
            print(
                f"\nWARNING: specular value or IoR value specified while using specular map")
    elif args.specular_ior:
        # FIXME: Does this need to be scaled properly?
        pbsdf_shader.inputs["Specular"].default_value = spec_from_ior(
            args.specular_ior)
    elif args.specular:
        pbsdf_shader.inputs["Specular"].default_value = args.specular
    else:
        pbsdf_shader.inputs["Specular"].default_value = 0.5

    print(f"\n")

    texture_nodes = []
    disp_node = None
    normal_node = None
    roughness_node = None

    # FIXME: We'd probably be better off loading a node tree and editing
    # it, rather than creating it from scratch.
    for sname, sdata in sockets.items():
        # DISPLACEMENT NODES
        # FIXME: This was separate in node wrangler, not sure why
        if sname == 'Displacement':
            disp_texture = nodes.new(type='ShaderNodeTexImage')
            # img = bpy.data.images.load(path.join(import_path, sname[2]))
            # img = bpy.data.images.load(os.path.join(os.getcwd(), sname[2]))
            img = bpy.data.images.load(os.path.realpath(sdata[1]))
            disp_texture.image = img
            disp_texture.label = 'Displacement'
            if disp_texture.image:
                disp_texture.image.colorspace_settings.is_data = True

            # Add displacement offset nodes
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.inputs["Scale"].default_value = args.height_scale
            disp_node.location = disp_texture.location + Vector((300, 0))
            link = links.new(disp_node.inputs[0], disp_texture.outputs[0])

            # could be 'DISPLACEMENT', 'BUMP' or 'BOTH'
            bpy.data.materials['Preview Material'].cycles.displacement_method = 'BOTH'

            # Find output node
            output_node = [n for n in nodes if n.bl_idname ==
                           'ShaderNodeOutputMaterial']
            if output_node:
                if not output_node[0].inputs[2].is_linked:
                    link = links.new(
                        output_node[0].inputs[2], disp_node.outputs[0])

            texture_nodes.append(disp_texture)
            continue

        # EVERYTHING ELSE
        if not pbsdf_shader.inputs[sname].is_linked:
            # No texture node connected -> add texture node with new image
            texture_node = nodes.new(type='ShaderNodeTexImage')
            img = bpy.data.images.load(os.path.realpath(sdata[1]))

            texture_node.image = img

            # NORMAL NODES
            if sname == 'Normal':
                # Test if new texture node is normal or bump map
                fname_components = split_into_components(sdata[1])
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
                    pbsdf_shader.inputs[sname], normal_node.outputs[0])
                normal_node_texture = texture_node

            elif sname == 'Roughness':
                # Test if glossy or roughness map
                fname_components = split_into_components(sdata[1])
                match_rough = set(tags["rough"]).intersection(
                    set(fname_components))
                match_gloss = set(tags["gloss"]).intersection(
                    set(fname_components))

                if match_rough:
                    # If Roughness nothing to do
                    link = links.new(
                        pbsdf_shader.inputs[sname], texture_node.outputs[0])

                elif match_gloss:
                    # If Gloss Map add invert node
                    invert_node = nodes.new(type='ShaderNodeInvert')
                    link = links.new(
                        invert_node.inputs[1], texture_node.outputs[0])

                    link = links.new(
                        pbsdf_shader.inputs[sname], invert_node.outputs[0])
                    roughness_node = texture_node
            else:
                # This is a simple connection Texture --> Input slot
                link = links.new(
                    pbsdf_shader.inputs[sname], texture_node.outputs[0])

            # Use non-color for all but 'Base Color' Textures
            if not sname in ['Base Color'] and texture_node.image:
                texture_node.image.colorspace_settings.is_data = True

        else:
            # If already texture connected. add to node list for alignment
            texture_node = pbsdf_shader.inputs[sname].links[0].from_node

        # This are all connected texture nodes
        texture_nodes.append(texture_node)
        texture_node.label = sname

    # We do this here, rather than way up there, so this will be last
    if disp_node:
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

    if disp_node:
        disp_node.location = disp_texture.location + Vector((300, 0))

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
    # Can use this if we'd rather not have a reroute node
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


output_re = re.compile(r"^(.*)\.([^.]+)", re.IGNORECASE)

def render(args, outfile):
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = 'CUDA'

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.feature_set = 'SUPPORTED'
    scene.render.resolution_percentage = args.scale
    scene.render.filepath = os.path.realpath(outfile)
    scene.cycles.use_denoising = args.denoise
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    # scene.cycles.denoiser = 'OPTIX'
    scene.cycles.samples = args.samples
    scene.cycles.use_progressive_refine = False

    scene.cycles.tile_order = 'CENTER'

    # Really, these scenes are simple enough that GPU vs. CPU, and one tile
    # size vs another probably doesn't matter. But lets set 'em anyhow.
    if args.cpu:
        scene.cycles.device = 'CPU'
        scene.render.tile_x = 32
        scene.render.tile_y = 32
    else:
        scene.cycles.device = 'GPU'
        scene.render.tile_x = 256
        scene.render.tile_y = 256

    # print(f"tile x: {scene.render.tile_x}   y: {scene.render.tile_y}")

    m = output_re.search(outfile)
    basename = m.group(1)
    filetype = str(m.group(2))

    if filetype == "png":
        scene.render.image_settings.file_format = "PNG"
    elif filetype == "jpg" or filetype == "jpeg":
        scene.render.image_settings.file_format = "JPEG"
    else:
        print(f"ERROR: unknown file extension for {os.path.basename(outfile)}")
        return

    print(
        f"INFO: output to render with size={scene.render.resolution_percentage}%,"
        f" denoising={args.denoise}, samples={args.samples}"
    )

    # FIXME: Should this be in main or somewhere else that's not here?
    if args.keep_blend:
        bpy.ops.wm.save_mainfile(filepath=basename + ".blend")

    if args.no_render:
        print("WARNING: Not rendering preview image due to --no-render")
    else:
        # FIXME: benchmark/validate the automatic tile size bit
        bpy.ops.preferences.addon_enable(module="render_auto_tile_size")
        bpy.ops.render.render(
            animation=False, write_still=True, use_viewport=False)


def readable_file(f: str) -> str:
    if not os.path.isfile(f) or not os.access(f, os.R_OK):
        raise argparse.ArgumentError(f"'{f}' is not a readable file")

    return f

# class SceneFileAction(argparse.Action):
#     def __call__(self, parser, ns, values, option):
#         setattr(ns, self.dest, option)

# Find the scene file. First, check the texrender scene directory for a
# filename that matches a plain scene name. Otherwise, assume a filename
# and check a couple of places for it.
def scene_file(f: str) -> str:
    if f.endswith(".blend"):
        return readable_file(f)

    # Not a blend file, so see if its in our scene directory
    mydir = os.path.realpath(os.path.dirname(__file__))
    scenepath = os.path.join(mydir, "scenes", f"texrender_scene_{f}.blend")
    if not os.path.isfile(scenepath):
        raise argparse.ArgumentError(f"scene name '{f}' is not valid")

    return scenepath


# graphic_re = re.compile(r"\.(exr|hdri?|jpe?g|png|tga|tiff?)$", re.IGNORECASE)
graphic_re = re.compile(r"\.png$", re.IGNORECASE)

# This edits 'args' in place, rather than returning something. Kind
# of (very) ugly.
def dirmode_prep(args):
    if len(args.files) != 1:
        print("ERROR: Directory mode requires exactly one input directory")
        sys.exit(1)

    d = args.files[0]
    if not os.path.isdir(d):
        print(f"ERROR: {d} does not exist, or is not a directory")
        sys.exit(1)

    # iterate the directory, only go one layer deep, and snag only files
    input_files = []
    files = os.listdir(d)
    for f in files:
        m = graphic_re.search(f)
        if not m:
            continue

        input_files.append(os.path.join(d, f))

    if len(input_files) == 0:
        print(f"ERROR: No graphics files found in {d}")
        sys.exit(1)

    args.files = input_files
    return


class NegateAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:4] != 'no')


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        # FIXME: We need to specify this, because we have no meaningful
        # argv[0], but we probably shouldn't just hardcode it
        prog='texrender.py',
        description='Render a preview of a (PBR) texture',
    )

    parser.add_argument("--debug", action="count", default=0)

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        # default="preview.png",
        default=None,

        help="file to output material preview to",
    )

    parser.add_argument(
        "-s",
        "--scene",
        help="blender scene file to use for rendering",
        type=scene_file,
        default="sphere",
    )

    parser.add_argument(
        "-sc",
        "--scale",
        help="scale scene by x%%",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--cpu",
        help="render with CPU instead of GPU",
        default=False,
        action='store_true',
    )

    # FIXME: Make these next two mutually exclusive
    parser.add_argument(
        "-sp",
        "--specular",
        default=None,
        type=float,
        help="flat specular value to be assigned to material (generally 0.00 - 0.08)",
    )

    parser.add_argument(
        "-si",
        "--specular-ior",
        default=None,
        type=float,
        help="calculate specular value to be assigned to material from material IoR",
    )

    parser.add_argument(
        "--height-scale",
        "--height",
        default=0.15,
        type=float,
        help="Scale to use for height (displacement) maps",
    )

    parser.add_argument(
        "-sa",
        "--samples",
        default=16,
        type=int,
        help="number of samples to use when rendering",
    )

    parser.add_argument(
        "--denoise",
        "--no-denoise",
        dest="denoise",
        default=True,
        action=NegateAction,
        nargs=0,
    )

    parser.add_argument(
        "-d",
        "--directory",
        default=False,
        action='store_true',
        help="directory mode",
    )

    parser.add_argument(
        "--analyze",
        default=False,
        action='store_true',
        help="show what maps would be used (no rendering)",
    )

    parser.add_argument(
        "-k",
        "--keep-blend",
        default=False,
        action='store_true',
        help="save a copy of the generated .blend file",
    )

    parser.add_argument(
        "-n",
        "--no-render",
        default=False,
        action='store_true',
        help="prep blend, but don't rendder (implies --keep-blend)",
    )

    parser.add_argument(
        "files",
        help="specify files to process",
        metavar="file",
        type=str,  # FIXME: Is there a 'file' type arg?
        nargs="+",
    )

    parsed_args = parser.parse_args(argv)

    # FIXME: This should probably be elsewhere? Especially if
    # we're going to support multiple input directories for
    # directory mode
    if parsed_args.out is None:
        if parsed_args.directory:
            d = os.path.dirname(parsed_args.files[0])
            parsed_args.out = f"{d}.png"
        else:
            parsed_args.out = "preview.png"

    return parsed_args


def main(argv: List[str]) -> int:
    print("texrender version: %s" % (git_version()))

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

    args = parse_arguments(argv)

    global _debug
    _debug = args.debug

    # directory mode
    # FIXME: Can we split this out somehow?
    if args.directory:
        # RIght now, only accept a single positional argument
        # FIXME: Accept more than one directory!
        dirmode_prep(args)

    # Theoretically you could specify --no-render and then not specify
    # --keep-blend, but at that point there's not really a point, so
    # go ahead and assume --keep-blend
    if args.no_render:
        args.keep_blend = True

    if not scene_prep(args, args.files):
        print("ERROR: Scene prep failed")
        return 1

    if args.analyze:
        return 0

    render(args, args.out)

    return 0


if __name__ == "__main__":
    ret = main(sys.argv)

    if ret != 0:
        # FIXME: How *do* we want to handle failures?
        print(f"WARNING: texrender exiting with return code {ret}")

    bpy.ops.wm.quit_blender()

    # Should never be reached
    sys.exit(0)
