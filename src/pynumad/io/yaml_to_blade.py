import yaml
import numpy as np
import logging
from scipy.stats import mode

from pynumad.utils.misc_utils import (
    LARCetaT,
    LARCetaL,
    _parse_data,
    full_keys_from_substrings,
)
from pynumad.io.airfoil_to_xml import airfoil_to_xml
from pynumad.utils.interpolation import interpolator_wrap
from pynumad.objects.component import Component
from pynumad.objects.airfoil import Airfoil
from pynumad.objects.material import Material
from pynumad.objects.definition import Definition


def yaml_to_blade(blade, filename: str, write_airfoils: bool = False):
    """
    This method writes blade information from a .yaml file to a Blade object.
    The yaml file is expected to be formatted according to the WindIO ontology.
    See https://windio.readthedocs.io/en/stable/source/turbine.html.

    Parameters
    ----------
    blade : Blade
    filename : string
        path to .yaml file
    write_airfoils : bool
        Set true to write airfoil files while reading in data. Defaults to false.

    Returns
    -------
    blade : Blade
        input blade object populated with yaml data
    """

    # Read in yaml file as a nested dictionary
    with open(filename) as blade_yaml:
        # data = yaml.load(blade_yaml,Loader=yaml.FullLoader)
        data = yaml.load(blade_yaml, Loader=yaml.Loader)

    # initialize definition
    definition = Definition()
    blade.definition = definition

    # Obtain blade outer shape bem
    blade_outer_shape_bem = data["components"]["blade"]["outer_shape_bem"]

    # obtain hub outer shape bem
    try:
        hub_outer_shape_bem = data["components"]["hub"]["outer_shape_bem"]
    except KeyError:
        # older versions of wind ontology do not have 'outer_shape_bem' subsection for hub data
        hub_outer_shape_bem = data["components"]["hub"]

    # obtain blade internal structure
    blade_internal_structure = data["components"]["blade"]["internal_structure_2d_fem"]

    # obtain airfoil data
    af_data = data["airfoils"]

    # obtain material data
    mat_data = data["materials"]

    ### STATIONS / AIRFOILS
    _add_stations(
        definition,
        blade_outer_shape_bem,
        hub_outer_shape_bem,
        af_data,
        filename,
        write_airfoils,
    )
    ### MATERIALS
    _add_materials(definition, mat_data)

    ## Blade Components

    # Update "grid" and "values" keys to cover the whole span of the blade
    blade_internal_structure = update_internal_structure(
        blade_internal_structure, blade_outer_shape_bem
    )

    blade_structure_dict = {
        blade_internal_structure["layers"][i]["name"].lower(): blade_internal_structure[
            "layers"
        ][i]
        for i in range(len(blade_internal_structure["layers"]))
    }
    # Spar caps
    _add_spar_caps(definition, blade_structure_dict)

    # TE Bands
    _add_te_bands(definition, blade_structure_dict)

    # LE Bands
    _add_le_bands(definition, blade_structure_dict)

    ### COMPONENTS
    _add_components(definition, blade_internal_structure, blade_structure_dict)

    blade.update_blade()
    # save(blade_name)
    # BladeDef_to_NuMADfile(obj,numad_name,matdb_name,numad_af_folder)
    return blade


def _add_stations(
    definition,
    blade_outer_shape_bem,
    hub_outer_shape_bem,
    af_data,
    file: str,
    write_airfoils,
):
    # Obtaining some parameters not explicitly given in YAML file
    L = np.ceil(blade_outer_shape_bem["reference_axis"]["z"]["values"][-1])
    R = L + hub_outer_shape_bem["diameter"] / 2
    L = R - hub_outer_shape_bem["diameter"] / 2
    definition.span = np.multiply(
        np.transpose(blade_outer_shape_bem["reference_axis"]['z']['grid']), L
    )
    definition.ispan = definition.span

    # Aerodynamic properties
    # using interp because yaml can have different r/R for twist and chord
    temp_x = np.transpose(blade_outer_shape_bem["twist"]["grid"])
    temp_y = blade_outer_shape_bem["twist"]["values"]
    definition.degreestwist = (
        interpolator_wrap(np.multiply(temp_x, L), np.transpose(temp_y), definition.span)
        * 180.0
        / np.pi
    )
    definition.chord = interpolator_wrap(
        np.multiply(np.transpose(blade_outer_shape_bem["chord"]["grid"]), L),
        np.transpose(blade_outer_shape_bem["chord"]["values"]),
        definition.span,
    )
    af_dir_names = []
    for i in range(len(af_data)):
        af_dir_names.append(af_data[i]["name"])
    numstations = len(blade_outer_shape_bem["airfoil_position"]["labels"])
    tc = [None] * numstations
    aero_cent = [None] * numstations
    definition.stations = []
    for i in range(numstations):
        _, _, iaf_temp = np.intersect1d(
            blade_outer_shape_bem["airfoil_position"]["labels"][i],
            af_dir_names,
            "stable",
            return_indices=True,
        )
        IAF = iaf_temp[0]  # Expect only one index of intersection
        tc[i] = af_data[IAF]["relative_thickness"]
        tc_xL = blade_outer_shape_bem["airfoil_position"]["grid"][i]
        aero_cent[i] = af_data[IAF]["aerodynamic_center"]
        x = np.array(af_data[IAF]["coordinates"]["x"], dtype=float)
        y = np.array(af_data[IAF]["coordinates"]["y"], dtype=float)
        xf_coords = np.stack((x, y), 1)

        # find coordinate direction (clockwise or counter-clockwise) Winding
        # Number. clockwise starting at (1,0) is correct
        with np.errstate(divide="ignore", invalid="ignore"):
            if (
                np.nanmean(np.gradient(np.arctan(xf_coords[:, 1] / xf_coords[:, 0])))
                > 0
            ):
                xf_coords = np.flipud(xf_coords)

        if write_airfoils:
            import os

            out_folder = "yaml2BladeDef_" + file.replace(".yaml", "")
            # blade_name = out_folder + '/' + file.replace('.yaml','') + '_definition.mat'
            # matdb_name =...
            # numade_name =...

            # Creating folders
            os.makedirs(out_folder + "/af_coords/", exist_ok=True)
            # os.makedirs(out_folder+'/af_polars/', exist_ok = True)
            os.makedirs(out_folder + "/airfoil/", exist_ok=True)
            airfoil_to_xml(
                xf_coords,
                blade_outer_shape_bem["airfoil_position"]["labels"][i],
                out_folder
                + "/af_coords/"
                + blade_outer_shape_bem["airfoil_position"]["labels"][i]
                + ".txt",
            )

        ref = blade_outer_shape_bem["airfoil_position"]["labels"][i]
        af = Airfoil(coords=xf_coords, reference=ref)
        af.resample(spacing="half-cosine")
        definition.add_station(af, tc_xL * L)

    definition.percentthick = np.multiply(
        interpolator_wrap(
            np.multiply(blade_outer_shape_bem["airfoil_position"]["grid"], L),
            tc,
            definition.span,
        ),
        100,
    )
    definition.aerocenter = interpolator_wrap(
        np.multiply(blade_outer_shape_bem["airfoil_position"]["grid"], L),
        aero_cent,
        definition.span,
    )
    definition.chordoffset = interpolator_wrap(
        np.multiply(np.transpose(blade_outer_shape_bem["pitch_axis"]["grid"]), L),
        np.transpose(blade_outer_shape_bem["pitch_axis"]["values"]),
        definition.span,
    )
    definition.natural_offset = 0
    definition.prebend = interpolator_wrap(
        np.multiply(
            np.transpose(blade_outer_shape_bem["reference_axis"]["x"]["grid"]), L
        ),
        np.transpose(blade_outer_shape_bem["reference_axis"]["x"]["values"]),
        definition.span,
    )
    definition.sweep = interpolator_wrap(
        np.multiply(
            np.transpose(blade_outer_shape_bem["reference_axis"]["y"]["grid"]), L
        ),
        np.transpose(blade_outer_shape_bem["reference_axis"]["y"]["values"]),
        definition.span,
    )

    # for i in range(len(tc)):
    #     afc = AirfoilDef(out_folder +
    #     '/af_coords/' +
    #     blade_outer_shape_bem['airfoil_position']['labels'][i] +
    #     '.txt')
    #     definition.add_station(afc,np.multiply(tc_xL[i],L))

    # NOTE nothing happens to afc? Tentatively ignoring...
    # If i return to this make sure to listify the afcs
    ### AIRFOILS
    # for i in range(len(tc)):
    #     afc = AirfoilDef(out_folder + '/af_coords/' +
    #         blade_outer_shape_bem['airfoil_position']['labels'][i] +
    #         '.txt')
    #     definition.add_station(afc,np.multiply(tc_xL[i],L))
    # afc.resample #NOTE afc isn't used after this... why resample?
    return


def _add_materials(definition, material_data):
    materials_dict = dict()
    for i in range(len(material_data)):
        cur_mat = Material()
        cur_mat.name = material_data[i]["name"]
        if material_data[i]["orth"] == 1:
            cur_mat.type = "orthotropic"
        else:
            cur_mat.type = "isotropic"
        # Add ply thickness option if ply thickness exists in yaml
        try:
            cur_mat.layerthickness = material_data[i]["ply_t"] * 1000
        except KeyError:
            msg = "material ply thickness " + \
            material_data[i]["name"] + \
            "not defined, assuming 1 mm thickness"
            logging.debug(msg)
            cur_mat.layerthickness = 1

        finally:
            pass

        # first
        cur_mat.uts = _parse_data(material_data[i]["Xt"])
        cur_mat.ucs = -_parse_data(material_data[i]["Xc"])
        cur_mat.uss = _parse_data(material_data[i]["S"])
        cur_mat.xzit = 0.3
        cur_mat.xzic = 0.25
        cur_mat.yzit = 0.3
        cur_mat.yzic = 0.25
        try:
            cur_mat.g1g2 = material_data[i].get("GIc", 0) / material_data[i].get(
                "GIIc", 0
            )
        except ZeroDivisionError:
            cur_mat.g1g2 = np.nan
        if "alp0" in material_data[i]:
            cur_mat.alp0 = _parse_data(material_data[i]["alp0"])
            cur_mat.etat = LARCetaT(cur_mat.alp0)
        else:
            cur_mat.alp0 = None
            cur_mat.etat = None
        try:
            # test if property is a list
            material_data[i]["E"] + []
        except TypeError:
            cur_mat.ex = _parse_data(material_data[i]["E"])
            cur_mat.ey = _parse_data(material_data[i]["E"])
            cur_mat.ez = _parse_data(material_data[i]["E"])
            cur_mat.gxy = _parse_data(material_data[i]["G"])
            cur_mat.gxz = _parse_data(material_data[i]["G"])
            cur_mat.gyz = _parse_data(material_data[i]["G"])
            cur_mat.prxy = _parse_data(material_data[i]["nu"])
            cur_mat.prxz = _parse_data(material_data[i]["nu"])
            cur_mat.pryz = _parse_data(material_data[i]["nu"])
            cur_mat.etal = LARCetaL(cur_mat.uss, cur_mat.ucs, cur_mat.alp0)
        else:
            cur_mat.ex = _parse_data(material_data[i]["E"][0])
            cur_mat.ey = _parse_data(material_data[i]["E"][1])
            cur_mat.ez = _parse_data(material_data[i]["E"][2])
            cur_mat.gxy = _parse_data(material_data[i]["G"][0])
            cur_mat.gxz = _parse_data(material_data[i]["G"][1])
            cur_mat.gyz = _parse_data(material_data[i]["G"][2])
            cur_mat.prxy = _parse_data(material_data[i]["nu"][0])
            cur_mat.prxz = _parse_data(material_data[i]["nu"][1])
            cur_mat.pryz = _parse_data(material_data[i]["nu"][2])
            cur_mat.etal = LARCetaL(cur_mat.uss[0], cur_mat.ucs[1], cur_mat.alp0)
        try:
            cur_mat.m = material_data[i]["m"]
        except KeyError:
            msg = f"No fatigue exponent found for material: {material_data[i]['name']}"
            logging.debug(msg)
        cur_mat.density = material_data[i]["rho"]
        # cur_mat.dens = mat_data[i]['rho']
        cur_mat.drydensity = material_data[i]["rho"]
        if (
            "description" in material_data[i].keys()
            and "source" in material_data[i].keys()
        ):
            desc_sourc = [
                material_data[i]["description"],
                ", ",
                material_data[i]["source"],
            ]
            cur_mat.reference = "".join(desc_sourc)
        else:
            cur_mat.reference = []

        materials_dict[cur_mat.name] = cur_mat
    definition.materials = materials_dict
    return


def _add_components(definition, blade_internal_structure, blade_structure_dict):
    N_layer_comp = len(blade_internal_structure["layers"])
    component_list = list()
    for i in range(N_layer_comp):
        i_component_data = blade_internal_structure["layers"][i]
        cur_comp = Component()
        cur_comp.group = 0
        cur_comp.name = i_component_data["name"]
        #   comp['material'] = blade_internal_structure['layers']{i}['material'];
        # mat_names = [mat.name for mat in definition.materials]
        # C,IA,IB = np.intersect1d(mat_names,i_component_data['material'],return_indices=True)
        cur_comp.materialid = i_component_data["material"]
        try:
            cur_comp.fabricangle = np.mean(
                i_component_data["fiber_orientation"]["values"]
            )
        finally:
            pass
        if "spar" in i_component_data["name"].lower():
            cur_comp.imethod = "pchip"
        else:
            cur_comp.imethod = "linear"
        # cur_comp.control_points[:,0] = np.transpose(i_component_data['thickness']['grid'])
        cptemp1 = np.transpose(i_component_data["thickness"]["grid"])
        temp_n_layer = (
            np.multiply(np.transpose(i_component_data["thickness"]["values"]), 1000.0)
            / definition.materials[cur_comp.materialid].layerthickness
        )
        I_round_up = np.flatnonzero((temp_n_layer > 0.05) & (temp_n_layer < 0.5))
        cptemp2 = np.round(
            np.multiply(np.transpose(i_component_data["thickness"]["values"]), 1000.0)
            / definition.materials[cur_comp.materialid].layerthickness
        )
        cur_comp.control_points = np.stack((cptemp1, cptemp2), axis=1)
        # if I_round_up.size > 0:
        #     cur_comp.control_points[I_round_up,1] = 1 # increase n_layers from 0 to 1 for 0.05<n_layers<0.5
        #     comp['cp'](:,2) = cell2mat(blade_internal_structure['layers']{i}['thickness']['values'])'.*1000;  # use when each material ply is 1 mm
        cur_comp.pinnedends = 0
        component_list.append(cur_comp)

    component_dict = dict()
    for comp in component_list:
        component_dict[comp.name] = comp

    # Spar Caps (pressure and suction)
    key_list = full_keys_from_substrings(component_dict.keys(), ["spar", "ps"])
    component_dict[key_list[0]].hpextents = ["b", "c"]

    key_list = full_keys_from_substrings(component_dict.keys(), ["spar", "ss"])
    component_dict[key_list[0]].lpextents = ["b", "c"]

    # uv coating
    key_list = full_keys_from_substrings(component_dict.keys(), ["uv"])  # Try 1
    if len(key_list) == 0:
        key_list = full_keys_from_substrings(component_dict.keys(), ["gel"])  # Try 2

    if len(key_list) == 1:
        component_dict[key_list[0]].hpextents = ["le", "te"]
        component_dict[key_list[0]].lpextents = ["le", "te"]
    elif len(key_list) == 0:
        raise ValueError("No UV or gelcoat found")
    else:
        raise ValueError("Too many uv or gelcoat components")

    # Shell skin
    key_list = full_keys_from_substrings(component_dict.keys(), ["shell"])
    if len(key_list) == 2:
        component_dict[key_list[0]].hpextents = ["le", "te"]
        component_dict[key_list[0]].lpextents = ["le", "te"]
        component_dict[key_list[1]].hpextents = ["le", "te"]
        component_dict[key_list[1]].lpextents = ["le", "te"]
    else:
        raise ValueError("Incorrect number of shell components")

    # TE Band(s)
    key_list = full_keys_from_substrings(component_dict.keys(), ["te", "reinf"])
    if len(key_list) == 1:
        component_dict[key_list[0]].hpextents = ["d", "te"]
        component_dict[key_list[0]].lpextents = ["d", "te"]
    elif len(key_list) == 2:
        tempKeyList = full_keys_from_substrings(key_list, ["ss"])
        if len(tempKeyList) == 1:
            component_dict[tempKeyList[0]].lpextents = ["d", "te"]
        else:
            ValueError("Incorrect number of te reinf ss components")

        tempKeyList = full_keys_from_substrings(key_list, ["ps"])
        if len(tempKeyList) == 1:
            component_dict[tempKeyList[0]].hpextents = ["d", "te"]
        else:
            ValueError("Incorrect number of te reinf ps components")
    else:
        raise ValueError("Invalid number of LE reinforcements")

    # LE Band(s)
    key_list = full_keys_from_substrings(component_dict.keys(), ["le", "reinf"])
    if len(key_list) == 1:
        component_dict[key_list[0]].hpextents = ["le", "a"]
        component_dict[key_list[0]].lpextents = ["le", "a"]
    elif len(key_list) == 2:
        tempKeyList = full_keys_from_substrings(key_list, ["ss"])
        if len(tempKeyList) == 1:
            component_dict[tempKeyList[0]].lpextents = ["le", "a"]
        else:
            ValueError("Incorrect number of te reinf ss components")

        tempKeyList = full_keys_from_substrings(key_list, ["ps"])
        if len(tempKeyList) == 1:
            component_dict[tempKeyList[0]].hpextents = ["le", "a"]
        else:
            ValueError("Incorrect number of te reinf ps components")
    else:
        raise ValueError("Invalid number of LE reinforcements")

    # Trailing edge suction-side panel
    key_list = full_keys_from_substrings(component_dict.keys(), ["te_", "ss", "filler"])
    if len(key_list) == 1:
        component_dict[key_list[0]].lpextents = ["c", "d"]
    else:
        raise ValueError("Invalid number of trailing edge suction-side panels")

    # Leading edge suction-side panel
    key_list = full_keys_from_substrings(component_dict.keys(), ["le_", "ss", "filler"])
    if len(key_list) == 1:
        component_dict[key_list[0]].lpextents = ["a", "b"]
    else:
        raise ValueError("Invalid number of leading edge suction-side panels")

    # Trailing edge suction-side panel
    key_list = full_keys_from_substrings(component_dict.keys(), ["le_", "ps", "filler"])
    if len(key_list) == 1:
        component_dict[key_list[0]].hpextents = ["a", "b"]
    else:
        raise ValueError("Invalid number of leading edge pressure-side panels")

    # Leading edge suction-side panel
    key_list = full_keys_from_substrings(component_dict.keys(), ["te_", "ps", "filler"])
    if len(key_list) == 1:
        component_dict[key_list[0]].hpextents = ["c", "d"]
    else:
        raise ValueError("Invalid number of trailing edge pressure-side panels")

    # Web
    #
    # Group every shear-web layer by the YAML's ordered ``webs`` list so
    # the downstream ``StackDatabase.swstacks`` ends up with one row per
    # YAML-declared web (in YAML order), regardless of how many webs the
    # blade has.  The legacy implementation collapsed all webs into two
    # groups using substring matches on layer NAMES (``fore``/``aft`` or
    # the digits ``0``/``1``), which silently dropped a third web for
    # IEA-22 and was inherently fragile (e.g. a layer named
    # ``web2_skin00`` matched ``"0"`` and landed in the wrong group).
    #
    # Each shear-web layer in the YAML carries a ``web:`` field whose
    # value names one of the entries in ``webs`` (e.g. ``web0``,
    # ``fore_web``).  We use that as the authoritative grouping signal
    # and fall back to substring matching only when ``web:`` is missing.

    for comp in component_dict:
        logging.debug(comp)

    # 1) Ordered list of web names from the YAML.
    webs_in_order = [
        w["name"]
        for w in blade_internal_structure.get("webs", [])
    ]

    # Per-web chord-side extent labels.  For backward compatibility with
    # 2-web BAR-family blades the first two webs land on the spar-cap
    # edges (``b`` and ``c``); subsequent webs cycle through the
    # remaining design keypoints (``a`` LE-side panel and ``d`` TE-side
    # panel).  Single key-label extents keep ``KeyPoints.web_indices``
    # non-NaN so ``mesh_gen.shell_mesh_general`` actually emits SW
    # element sets for every web.  Webs at non-keypoint chordwise
    # positions (e.g. the IEA-22 ``start_nd_arc``/``end_nd_arc`` spec)
    # are placed approximately at these design keypoints — exact
    # placement from per-station arcs is not yet supported by the
    # keypoint/mesh pipeline, but at least all webs are now LOADED and
    # MESHED instead of silently dropped.
    _DEFAULT_WEB_EXTENTS = ["b", "c", "a", "d"]

    # 2) Map yaml-web-name -> group-index (1-based) preserving order.
    name_to_group = {
        name.lower(): (i + 1) for i, name in enumerate(webs_in_order)
    }

    # 3) Build a name-keyed lookup of layer dicts so we can read the
    #    per-layer ``web:`` field.  ``blade_structure_dict`` already
    #    contains every layer keyed by its lowercased name.
    found_groups = set()
    unassigned_keys = []
    for comp_name, comp in component_dict.items():
        layer = blade_structure_dict.get(comp_name.lower(), {})
        web_field = layer.get("web") if isinstance(layer, dict) else None
        if web_field and web_field.lower() in name_to_group:
            grp = name_to_group[web_field.lower()]
            web_idx = grp - 1
            extent = _DEFAULT_WEB_EXTENTS[web_idx % len(_DEFAULT_WEB_EXTENTS)]
            comp.hpextents = [extent]
            comp.lpextents = [extent]
            comp.group = grp
            found_groups.add(grp)
        else:
            unassigned_keys.append(comp_name)

    # 4) Legacy fallback: for layers without a usable ``web:`` field
    #    (older YAMLs), fall back to substring matching the layer name
    #    against ``fore``/``web1`` and ``aft``/``web0``/``rear``.  This
    #    preserves the original two-web behaviour bit-identically.
    if unassigned_keys:
        # Subset of component_dict containing only unassigned layers.
        leftover = {k: component_dict[k] for k in unassigned_keys}

        fore_keys = full_keys_from_substrings(leftover.keys(), ["web", "fore"])
        if len(fore_keys) == 0:
            fore_keys = full_keys_from_substrings(leftover.keys(), ["web", "1"])
        for key in fore_keys:
            component_dict[key].hpextents = ["b"]
            component_dict[key].lpextents = ["b"]
            component_dict[key].group = 1
            found_groups.add(1)

        rear_keys = full_keys_from_substrings(leftover.keys(), ["web", "aft"])
        if len(rear_keys) == 0:
            rear_keys = full_keys_from_substrings(leftover.keys(), ["web", "0"])
        if len(rear_keys) == 0:
            rear_keys = full_keys_from_substrings(leftover.keys(), ["web", "rear"])
        for key in rear_keys:
            component_dict[key].hpextents = ["c"]
            component_dict[key].lpextents = ["c"]
            component_dict[key].group = 2
            found_groups.add(2)

    # 5) Sanity-check that at least one web was identified.  If the YAML
    #    declared webs but no layer carried a ``web:`` field AND the
    #    substring fallback also matched nothing, that is a malformed
    #    YAML.
    web_layers_in_dict = any(
        ("web" in name.lower()) for name in component_dict.keys()
    )
    if web_layers_in_dict and not found_groups:
        raise ValueError(
            "No shear-web layers were grouped: YAML contains layers whose "
            "name includes 'web' but none carries a recognised 'web:' "
            "field and substring matching on the name found no fore/aft "
            "pattern."
        )

    ### add components to blade
    definition.components = component_dict
    return


def update_internal_structure(blade_internal_structure, blade_outer_shape_bem):
    bladeParts = ["layers", "webs"]
    # Make sure each definition.ispan has layer thicknesses and widths
    fullSpanGrid = np.array(blade_outer_shape_bem["reference_axis"]['z']['grid'])
    nStations = len(fullSpanGrid)
    keysToModify = {
        "offset_y_pa",
        "thickness",
        "fiber_orientation",
        "width",
        "start_nd_arc",
        "end_nd_arc",
    }
    for part_name in bladeParts:
        N_layer_comp = len(blade_internal_structure[part_name])
        for currentLayer in range(N_layer_comp):
            layerKeys = set(blade_internal_structure[part_name][currentLayer].keys())

            for currentKey in keysToModify.intersection(layerKeys):
                grid = blade_internal_structure[part_name][currentLayer][currentKey][
                    "grid"
                ]
                values = blade_internal_structure[part_name][currentLayer][currentKey][
                    "values"
                ]
                startStationLoc = grid[0]
                endStationLoc = grid[-1]

                subSpanGridIndex = np.where(
                    (fullSpanGrid >= startStationLoc) & (fullSpanGrid <= endStationLoc)
                )[0]

                # iterpolate fullSpanGrid locations onto layer grid defined in the yamle file for the layer
                subSpanValues = interpolator_wrap(
                    grid, values, fullSpanGrid[subSpanGridIndex], "pchip"
                )
                fullSpanValues = np.zeros(nStations)

                fullSpanValues[subSpanGridIndex] = subSpanValues

                # Reset
                blade_internal_structure[part_name][currentLayer][currentKey][
                    "grid"
                ] = fullSpanGrid
                blade_internal_structure[part_name][currentLayer][currentKey][
                    "values"
                ] = fullSpanValues
    return blade_internal_structure


def _add_spar_caps(definition, blade_structure_dict):
    sparCapKeys = full_keys_from_substrings(blade_structure_dict.keys(), ["spar"])
    if len(sparCapKeys) != 2:
        raise ValueError("Incorrect number of spar cap components")

    lpSideIndex = None
    hpSideIndex = None
    for iSparCap in range(2):
        if "suc" in blade_structure_dict[sparCapKeys[iSparCap]]["side"].lower():
            lpSideIndex = iSparCap
        if "pres" in blade_structure_dict[sparCapKeys[iSparCap]]["side"].lower():
            hpSideIndex = iSparCap
    if lpSideIndex is None or hpSideIndex is None:
        sides = [blade_structure_dict[k]["side"] for k in sparCapKeys]
        raise ValueError(
            f"Could not identify suction/pressure spar caps from sides "
            f"{sides!r}; expected substrings 'suc' and 'pres' in `side` fields."
        )

    definition.sparcapwidth_lp = (
        blade_structure_dict[sparCapKeys[lpSideIndex]]["width"]["values"] * 1000
    )
    try:
        definition.sparcapoffset_lp = (
            blade_structure_dict[sparCapKeys[lpSideIndex]]["offset_y_pa"]["values"]
            * 1000
        )
    except KeyError:
        definition.sparcap_start_nd_arc = blade_structure_dict[
            sparCapKeys[lpSideIndex]
        ]["start_nd_arc"]["values"]
        definition.sparcap_end_nd_arc = blade_structure_dict[sparCapKeys[lpSideIndex]][
            "end_nd_arc"
        ]["values"]

    definition.sparcapwidth_hp = (
        blade_structure_dict[sparCapKeys[hpSideIndex]]["width"]["values"] * 1000
    )
    try:
        definition.sparcapoffset_hp = (
            blade_structure_dict[sparCapKeys[hpSideIndex]]["offset_y_pa"]["values"]
            * 1000
        )
    except KeyError:
        definition.sparcap_start_nd_arc = blade_structure_dict[
            sparCapKeys[hpSideIndex]
        ]["start_nd_arc"]["values"]
        definition.sparcap_end_nd_arc = blade_structure_dict[sparCapKeys[hpSideIndex]][
            "end_nd_arc"
        ]["values"]
    return definition


def _add_te_bands(definition, blade_structure_dict):
    teReinfKeys = full_keys_from_substrings(blade_structure_dict.keys(), ["te", "reinf"])
    if len(teReinfKeys) == 1:
        definition.teband = (
            blade_structure_dict[teReinfKeys[0]]["width"]["values"] * 1000 / 2
        )
    elif len(teReinfKeys) == 2:
        definition.teband = (
            (
                blade_structure_dict[teReinfKeys[0]]["width"]["values"]
                + blade_structure_dict[teReinfKeys[1]]["width"]["values"]
            )
            * 1000
            / 2
        )
    else:
        raise ValueError("Unknown number of TE reinforcements")
    return definition


def _add_le_bands(definition, blade_structure_dict):
    leReinfKeys = full_keys_from_substrings(blade_structure_dict.keys(), ["le", "reinf"])
    if len(leReinfKeys) == 1:
        definition.leband = (
            blade_structure_dict[leReinfKeys[0]]["width"]["values"] * 1000 / 2
        )
    elif len(leReinfKeys) == 2:
        definition.leband = (
            (
                blade_structure_dict[leReinfKeys[0]]["width"]["values"]
                + blade_structure_dict[leReinfKeys[1]]["width"]["values"]
            )
            * 1000
            / 2
        )
    else:
        raise ValueError("Invalid number of LE reinforcements")
    return definition
