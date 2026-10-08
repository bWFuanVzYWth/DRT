# Blender AgX asset attribution and license record

Original source: https://github.com/blender/blender/tree/8b5bd560cf16070d10f361d0a9f00b39436dc379/release/datafiles/colormanagement

The pinned OpenColorIO config credits original AgX by Troy Sobotka, with further development by Zijun Eary Zhou, Mark Faderbauer and Sakari Kapanen, and links https://github.com/EaryChow/AgX.

The config states "See ocio-license.txt for details", but that file is not present in this pinned checkout. The LUT has author/config attribution but no separate license identifier in its header. Accordingly the third-party LUT license is recorded as **NOASSERTION**; this notice does not replace or invent the missing upstream terms, or relicense the original asset as MIT/GPL.

The workbench's own WGSL interpolation/input/output adapters are project code. Packed float32 values preserve the original grid; source and packed hashes are reproducible through references/validation/generate_lut_assets.py and lut_assets.json.
