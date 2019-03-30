import os
import sys
import networkx as nx
import shutil
import os
import string
import random
import copy
import collections as cl

import numpy as np
import scipy.spatial as sp
import json

import set_akselos_path
#import exo_data_tools as et
#import block_attribute_tools as bt
import akselos.mesh.ansys_parser as ap
# import akselos.mesh.inp_parser as ip
# import akselos.mesh.exo_data_tools as et
import new_type
# import selection_mode as sm
# import stored_selection as ss
# import face_ref as fr
import loads as ld
import mesh.mesh_port_tools as mt
import point as pt


import types_data as td
import directories

import rb_component_system as cs
import rb_components as rb
import json_helper
import interfaces.akselos_assembly as aa
import operator_id as oi
import operator_type as ot
import id_objects.port_id
import interfaces.types as tp
import dimensional_value as dv
import id_objects as cu
import plugins.plugin_interface as pi
import ui_objects.factory.ref_component_type_geom_ui as ru
import updater



API = pi.AkselosAPI()
sys.setrecursionlimit(50000)


COMMENT_LINE = "*"

NODE_KEY = "JOINT"
MEMBER_KEY = "MEMBER"
ELEM_GLOBAL_OFFSET_KEY = "MEMBER1"
ELEM_LOCAL_OFFSET_KEY = "MEMBER2"
ELEM_OFFSET_KEY = "MEMBER OFFSETS"
SHELL_KEY = "PLATE"
SECT_KEY = "SECT"
GRUP_KEY = "GRUP"
LOAD_KEY = "LOADCN"
LOAD_COMBINATION_KEY = "LCOMB"
GRUP_OV_KEY = "GRPOV"
MEM_OV_KEY = "MEMOV"
MGR_OV_KEY = "MGROV"
CDM_KEY = "CDM"
PHYSICAL_KEY = "LDOPT"

LABEL_LEN = 4
GLOBAL_X = np.array([1., 0., 0.])
GLOBAL_Y = np.array([0., 1., 0.])
GLOBAL_Z = np.array([0., 0., 1.])
BEAM_DEFAULT_NODES = np.array([[-0.5, 0., 0.], [0.5, 0., 0.]])
CREATE_LOADS = True
CREATE_SHELL_COMPONENTS = True
CREATE_BUOYANCY_LOAD = False
CREATE_ALL_SOURCE_FOR_STIFF = False
ENABLE_AVERAGE_DENSITY_FOR_STIFF = True

ISOLATED_SIZE = 0.12
IS_USE_RIGID = True
LOAD_COEFF = 1.05
STIFF_DENSITY_TOL = 10

AKSELOS_LENGTH_UNIT = 'm'
AKSELOS_MASS_UNIT = 'kg'
AKSELOS_FORCE_UNIT = 'N'
AKSELOS_YOUNG_MODULUS_UNIT = 'GPa'
AKSELOS_DEFAULT_XSECT_STANDARD = 'aisc13th'


AKSELOS_TO_SACS_T1_T2_GEOMETRY_ANGLE = {
    "predefined_beams/i_section": 90.,
    "predefined_beams/rectangular": 90.,
    "predefined_beams/rectangular_tube": 90.
}


def convert_units(unit_type, method):
    all_to_meter = {'mm': 0.001, 'cm': 0.01, 'dm': 0.1, 'in': 0.0254, 'm': 1.0}
    all_to_kg = {'tonne': 1000.0, 'kg': 1.0, 'g': 0.001}
    all_to_newton = {'N': 1.0, 'kN': 1000.0}
    all_to_Pa = {'Pa': 1.0, 'MPa': 1.0e6, 'GPa': 1.0e9}

    unit_dict = {'length': all_to_meter, 'mass': all_to_kg, 'force': all_to_newton, 'young_modulus': all_to_Pa}
    assert unit_type in unit_dict.keys(), " Do not support the unit type '{}'".format(unit_type)

    keyword = ' to '
    assert keyword in method, "Wrong format. It should be 'unit1 to unit2'".format(keyword)
    parts = method.replace('\n','').strip().split(keyword)
    input_unit_name = parts[0].strip()
    output_unit_name = parts[1].strip()

    assert input_unit_name in unit_dict[unit_type].keys(), " Do not support the unit '{}' in '{}' with method '{}'"\
        .format(input_unit_name, unit_type, method)
    assert output_unit_name in unit_dict[unit_type].keys(), " Do not support the unit '{}' in '{}' with method '{}'"\
        .format(output_unit_name, unit_type, method)

    return unit_dict[unit_type][input_unit_name]/unit_dict[unit_type][output_unit_name]


def scale_units(sacs_units):
    # Convert sacs's units to akselos's units
    global XSECT_DIMENSION_SCALING, LONGITUDINAL_DIMENSION_SCALING
    # for x-section dimensions, ...
    XSECT_DIMENSION_SCALING = convert_units('length', '{} to {}'.format(sacs_units['xsect_length'], AKSELOS_LENGTH_UNIT))
    # for member length, ...
    LONGITUDINAL_DIMENSION_SCALING = convert_units('length', '{} to {}'.format(sacs_units['longitudinal_length'], AKSELOS_LENGTH_UNIT))

    # Note: take into account contributions of length units on propertity units
    global MASS_SCALING, MASS_DENSITY_SCALING, FORCE_SCALING, MODULI_SCALING
    MASS_SCALING = convert_units('mass', '{} to {}'.format(sacs_units['mass'], AKSELOS_MASS_UNIT))
    MASS_DENSITY_SCALING = MASS_SCALING / np.power(LONGITUDINAL_DIMENSION_SCALING, 3.0)
    FORCE_SCALING = convert_units('force', '{} to {}'.format(sacs_units['force'], AKSELOS_FORCE_UNIT))
    # convert young modulus to Pa
    MODULI_SCALING = convert_units('force', '{} to {}'.format(sacs_units['force'], 'N')) \
                            / np.power(convert_units('length', '{} to {}'.format(sacs_units['xsect_length'], 'm')), 2.0)
    MODULI_SCALING = MODULI_SCALING \
                             * convert_units('young_modulus', '{} to {}'.format('Pa', AKSELOS_YOUNG_MODULUS_UNIT))
    MODULI_SCALING *= 1.0e3


def parse_sacs(sacs_filepath, sacs_units):
    print("> Convert units")
    scale_units(sacs_units)
    #
    print ("> Read SACS data:")
    with open(sacs_filepath) as f:
        content = f.read()
    " Remove null line and comment line "
    text_lines = content.split("\n")
    text_lines = [line for line in text_lines
                  if len(line.lstrip()) > 0 and line[0] != COMMENT_LINE]

    read_line_idxs = []
    print ("Cross-section Info")
    sect_infos = read_sect_infos(text_lines, read_line_idxs)
    print ("Group Info")
    grup_infos = read_grup_infos(text_lines, read_line_idxs, sect_infos)
    print ("Node Info")
    node_labels, node_coords, constraint_nodes, linear_spring_nodes = \
        read_nodes(text_lines, read_line_idxs)
    print ("Beam Info")
    beam_elem_infos, beam_offset_infos, beam_joint_labels = \
        read_beam_elem_info(text_lines, node_labels, read_line_idxs)

    print ("Load Info")
    load_infos, all_beams_have_load = read_load_infos(text_lines, read_line_idxs, beam_joint_labels)
    print ("Load combination Info")
    load_combination_infos = read_combination_infos(text_lines, read_line_idxs)
    print ("Group Override Info")
    grp_ov_infos = read_grp_ov_infos(text_lines, read_line_idxs)
    print ("Member Override Info")
    mem_ov_infos = read_mem_ov_infos(text_lines, node_labels, read_line_idxs)
    print ("Read Model Physical Parameters Info")
    water_density, seabed_elevation, water_elevation = read_model_physical_infos(text_lines, read_line_idxs)
    print ("Marine Growth Override Info")
    mgr_ov_infos = read_mgr_ov_infos(text_lines, read_line_idxs, seabed_elevation)
    print ("Drag and Inertia Coefficient Info")
    cdm_infos = read_cdm_infos(text_lines, read_line_idxs)
    #new_beam_infos = split_beams(beam_elem_infos, split_beam_infos, node_coords)

    # shell_elems = read_shell_elem_infos(text_lines, node_labels, read_line_idxs)
    # shell_component_elems = separate_shell_elems(shell_elems)
    # shell_exo_files, free_beam_elems, component_stiffener_info = write_shell_component_elems_to_exo_data(
    #     shell_component_elems, node_coords, beam_elem_infos)

    #component_stiffener_info = []

    model_info = {}
    #model_info['component_stiffener_info'] = component_stiffener_info
    model_info['sect_infos'] = sect_infos
    model_info['grup_infos'] = grup_infos
    model_info['load_infos'] = load_infos
    model_info['load_combination_infos'] = load_combination_infos
    model_info['node_labels'] = node_labels
    model_info['constraint_nodes'] = constraint_nodes
    model_info['linear_spring_nodes'] = linear_spring_nodes
    model_info['all_beams_have_load'] = all_beams_have_load
    model_info['beam_offset_infos'] = beam_offset_infos
    model_info['grp_ov_infos'] = grp_ov_infos
    model_info['mem_ov_infos'] = mem_ov_infos
    model_info['mgr_ov_infos'] = mgr_ov_infos
    model_info['water_density'] = water_density
    model_info['seabed_elevation'] = seabed_elevation
    model_info['water_elevation'] = water_elevation
    model_info['cdm_infos'] = cdm_infos

    return node_coords, beam_elem_infos, model_info


def read_sect_infos(text_lines, read_line_idxs):
    sect_infos = {}
    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue
        if not SECT_KEY == line_text[:len(SECT_KEY)]:
            continue
        if line_text[len(SECT_KEY):].strip() == '':
            # empty sect line
            continue
        #
        sidx, eidx = 5, 15
        sect_name = line_text[sidx:eidx].strip()
        sidx, eidx = eidx, 18
        beam_type = line_text[sidx:eidx].strip()
        # Geometric dimensions
        sidx = 49
        dimensions = []
        text_lens = [6, 5, 6, 5, 5, 4]
        for text_len in text_lens:
            eidx = sidx + text_len
            # TODO: review converting a string to a float number
            real_val = convert_str_to_float_with_default(line_text[sidx:eidx].strip()) * XSECT_DIMENSION_SCALING
            dimensions.append(real_val)
            sidx = eidx
        infos = (beam_type, dimensions)
        sect_infos[sect_name] = infos

        read_line_idxs.append(line_idx)

    return sect_infos


def read_grup_infos(text_lines, read_line_idxs, sect_infos):
    grup_infos = {}
    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not GRUP_KEY == line_text[:len(GRUP_KEY)]:
            continue

        if line_text[len(GRUP_KEY):].strip() == '':
            # empty group line
            continue

        grup_name = line_text[5:8].strip()
        sect_name = line_text[9:16].strip()
        dimensions = []
        if sect_name == "":
            # Single circular tube
            ro = float(line_text[17:23].strip())*XSECT_DIMENSION_SCALING
            t = float(line_text[23:29].strip())*XSECT_DIMENSION_SCALING
            dimensions = [ro, t]
        # TODO: consider unit for property paramaters
        young_modulus = float(line_text[30:35].strip()) * MODULI_SCALING
        shear_modulus = float(line_text[35:40].strip()) * MODULI_SCALING
        poisson_ratio = (young_modulus / (2.0 * shear_modulus)) - 1
        assert -1 < poisson_ratio < 0.5, "Invalid value of Poisson's ratio"
        if poisson_ratio < 0:
            # TODO: hack to pass gui's bug
            # Trong: change value to 0.3
            poisson_ratio = 0.3  # typical range of nu=(0.2, 0.5) for isotropic engineering materials
        flood_att = line_text[69].strip()
        if flood_att == 'N':
            flood_att = 0
        else:
            flood_att = 1


        mass_density = convert_to_float(line_text[70:76].strip()) * MASS_DENSITY_SCALING
        sub_beam_length = None
        if len(line_text.strip()) >= 79:
            sub_beam_length = convert_to_float(line_text[76:80].strip())*LONGITUDINAL_DIMENSION_SCALING

        if grup_name in grup_infos: # for segmented members (if a grup_name is read more than once)
            existing_grup_info = grup_infos[grup_name]
            if not isinstance(existing_grup_info, list): #if this is a tuple -> transform it to a list
                existing_grup_info = [existing_grup_info]

            new_grup_info = (sect_name, dimensions, young_modulus, mass_density, sub_beam_length, flood_att, poisson_ratio)
            existing_grup_info.append(new_grup_info)
            grup_infos[grup_name] = existing_grup_info
        else: # no segment members
            grup_info = (sect_name, dimensions, young_modulus, mass_density, sub_beam_length, flood_att, poisson_ratio)
            grup_infos[grup_name] = grup_info

        read_line_idxs.append(line_idx)

    return grup_infos


def read_load_infos(text_lines, read_line_idxs, beam_joint_labels):

    def can_add_partial_loads(current_load_values, load_values):
        if current_load_values[-1] != load_values[-1]: # load length
            return False
        if current_load_values[-2] != load_values[-2]: # load offset
            return False
        return True

    load_infos = {}
    load_name_to_load_index = {}  # if load name duplicate we increase this index
    all_beams_have_load = {} # Loop to read Main loadcases
    for line_idx, line_text_0 in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not LOAD_KEY == line_text_0[:len(LOAD_KEY)]:
            continue

        if line_text_0[len(LOAD_KEY):].strip() == '':
            # empty group line
            continue
        load_case_label = line_text_0[6:10].strip()
        read_line_idxs.append(line_idx)

        # LOADLB
        line_text_1 = text_lines[line_idx+1]
        line_text_2 = text_lines[line_idx+2]
        if line_text_2[:5] == "WAVE ":
            # TODO THUC: WAVE LOADING
            continue

        # assert line_text_1.startswith('LOADLB')
        _load_case_label = line_text_1[6:10].strip()
        # assert _load_case_label == load_case_label
        load_case_full_name = line_text_1[10:70].strip()
        read_line_idxs.append(line_idx+1)
        load_case_key = (load_case_label, load_case_full_name)

        if load_case_key not in load_infos:
            load_infos[load_case_key] = {}

        for line_idx in range(line_idx+2, len(text_lines)): # Loop to read sub-loadcases
            _line_text = text_lines[line_idx]
            if not text_lines[line_idx][:5] == "LOAD ":
                break

            if line_idx in read_line_idxs:
                continue

            if not _line_text.startswith('LOAD '):
                continue

            joint_0 = _line_text[7:11].strip()
            joint_1 = _line_text[11:15].strip()
            joint_labels = _line_text[7:15]
            assert len(joint_0) > 0

            beam_load_type = _line_text[65:69].strip()
            load_name = _line_text[72:80]

            if load_name not in load_infos[load_case_key]:
                load_infos[load_case_key][load_name] = {}

            if len(joint_1) > 0:
                # Load on Beams
                offset_0 = convert_str_to_float_with_default(_line_text[16:23].strip()) * LONGITUDINAL_DIMENSION_SCALING
                value_0 = convert_str_to_float_with_default(_line_text[23:30].strip()) * FORCE_SCALING
                load_length = convert_str_to_float_with_default(
                    _line_text[30:37].strip()) * LONGITUDINAL_DIMENSION_SCALING
                value_1 = convert_str_to_float_with_default(_line_text[37:44].strip()) * FORCE_SCALING
                load_coord_system = _line_text[60:64]
                load_direction = _line_text[5]

                if beam_load_type == "UNIF" and (offset_0 > 1e-4 or load_length > 1e-4):
                    beam_load_type = "PART-UNIF"

                # There are some member's load member has inverse joint labels
                if joint_labels not in beam_joint_labels:
                    joint_labels = joint_labels[4:] + joint_labels[:4]
                    assert joint_labels in beam_joint_labels
                    if beam_load_type in ["UNIF", "PART-UNIF"]:
                        _tmp = value_0
                        value_0 = value_1
                        value_1 = _tmp

                if beam_load_type == "UNIF":
                    # Load distributed on whole length of beam
                    load_values = \
                        [load_coord_system.upper(), 0., 0., 0., 0., 0., 0.]
                    if load_direction.upper() == 'X':
                        load_values[1] = value_0 / LONGITUDINAL_DIMENSION_SCALING
                        load_values[4] = value_1 / LONGITUDINAL_DIMENSION_SCALING
                    elif load_direction.upper() == 'Y':
                        load_values[2] = value_0 / LONGITUDINAL_DIMENSION_SCALING
                        load_values[5] = value_1 / LONGITUDINAL_DIMENSION_SCALING
                    elif load_direction.upper() == 'Z':
                        load_values[3] = value_0 / LONGITUDINAL_DIMENSION_SCALING
                        load_values[6] = value_1 / LONGITUDINAL_DIMENSION_SCALING
                    else:
                        assert False

                elif beam_load_type == "PART-UNIF":
                    # Load distributed on a part of length of beam
                    load_values = \
                        [load_coord_system.upper(), 0., 0., 0., 0., 0., 0.]
                    if load_direction.upper() == 'X':
                        load_values[1] = value_0 / LONGITUDINAL_DIMENSION_SCALING
                        load_values[4] = value_1 / LONGITUDINAL_DIMENSION_SCALING
                    elif load_direction.upper() == 'Y':
                        load_values[2] = value_0 / LONGITUDINAL_DIMENSION_SCALING
                        load_values[5] = value_1 / LONGITUDINAL_DIMENSION_SCALING
                    elif load_direction.upper() == 'Z':
                        load_values[3] = value_0 / LONGITUDINAL_DIMENSION_SCALING
                        load_values[6] = value_1 / LONGITUDINAL_DIMENSION_SCALING
                    else:
                        assert False

                    load_values.extend([offset_0, load_length])
                elif beam_load_type == "CONC":
                    # Load concentrated at a certain point of beam
                    load_values = [load_coord_system.upper(), offset_0, 0., 0., 0.]
                    if load_direction.upper() == 'X':
                        load_values[2] = value_0
                    elif load_direction.upper() == 'Y':
                        load_values[3] = value_0
                    elif load_direction.upper() == 'Z':
                        load_values[4] = value_0
                    else:
                        assert False

                else:
                    assert False, "Do not support this beam load type {}".format(beam_load_type)

                if not beam_load_type in all_beams_have_load:
                    all_beams_have_load[beam_load_type] = set([])

                load_item_key = (beam_load_type, joint_labels, load_direction)
                if load_item_key not in load_infos[load_case_key][load_name]:
                    load_infos[load_case_key][load_name][load_item_key] = tuple(load_values)
                else:
                    current_load_values = list(load_infos[load_case_key][load_name][load_item_key])
                    assert len(current_load_values) == len(load_values)
                    aggregate_load_values = copy.deepcopy(current_load_values)
                    if beam_load_type in ["UNIF"]:
                        for col_idx in range(1, 7):
                            aggregate_load_values[col_idx] += load_values[col_idx]
                    elif beam_load_type in ["PART-UNIF"]:
                        if can_add_partial_loads(current_load_values, load_values):
                            for col_idx in range(1, 7):
                                aggregate_load_values[col_idx] += load_values[col_idx]
                        else:

                            if load_name not in load_name_to_load_index:
                                load_name_to_load_index[load_name] = 1
                                duplicate_load_index = load_name_to_load_index[load_name]
                            else:
                                load_name_to_load_index[load_name] += 1
                                duplicate_load_index = load_name_to_load_index[load_name]

                            new_load_name = load_name + '-' + str(duplicate_load_index)
                            # Trong: fix bug of access before assignment
                            load_infos[load_case_key][new_load_name] = {}
                            load_infos[load_case_key][new_load_name][load_item_key] = tuple(
                                load_values)
                            print ("INFO: Rename load {} to {} because of duplicate name".format(
                                load_name, new_load_name))
                            all_beams_have_load[beam_load_type].add(joint_labels)
                            continue

                    elif beam_load_type == "CONC":
                        if load_name not in load_name_to_load_index:
                            load_name_to_load_index[load_name] = 1
                            duplicate_load_index = load_name_to_load_index[load_name]
                        else:
                            load_name_to_load_index[load_name] += 1
                            duplicate_load_index = load_name_to_load_index[load_name]

                        new_load_name = load_name + '-' + str(duplicate_load_index)
                        load_infos[load_case_key][new_load_name] = {}
                        load_infos[load_case_key][new_load_name][load_item_key] = tuple(
                            load_values)


                    # print ("INFO: found duplicate loads with same member, direction and load name", beam_load_type)
                    # print (current_load_values, load_values)
                    # print ('->', aggregate_load_values)
                    load_infos[load_case_key][load_name][load_item_key] = tuple(aggregate_load_values)

                all_beams_have_load[beam_load_type].add(joint_labels)

            else:
                # Load on Joints
                # Note: It includes 6 components for force and moment
                assert beam_load_type == "JOIN" # additional check
                load_type = "NODE"
                load_values = []
                # Note: 16, 7: default start idx and length
                for i in range(6):
                    val = convert_str_to_float_with_default(
                        _line_text[16+7*i:16+7*(i+1)].strip()) * FORCE_SCALING
                    load_values.append(val)
                    if i > 2:
                        # Moment components only
                        load_values[-1] *= LONGITUDINAL_DIMENSION_SCALING

                load_direction = None
                load_item_key = (load_type, joint_0, load_direction)

                if load_item_key not in load_infos[load_case_key][load_name]:
                    load_infos[load_case_key][load_name][load_item_key] = tuple(load_values)
                else:
                    current_load_values = list(load_infos[load_case_key][load_name][load_item_key])
                    # print ("INFO: found duplicate loads with same member, direction and load name")
                    # print (current_load_values)
                    # print (load_values)
                    assert len(current_load_values) == len(load_values)
                    aggregate_load_values = copy.deepcopy(current_load_values)
                    for col_idx in range(6):
                        aggregate_load_values[col_idx] += load_values[col_idx]
                    load_infos[load_case_key][load_name][load_item_key] = tuple(aggregate_load_values)

    return load_infos, all_beams_have_load


def read_model_physical_infos(text_lines, read_line_idxs):
    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not PHYSICAL_KEY == line_text[:len(PHYSICAL_KEY)]:
            continue

        if line_text[len(PHYSICAL_KEY):].strip() == '':
            # empty group line
            continue

        water_density = float(line_text[16:24].strip()) * 1000.0

        try:
            seabed_elevation = float(line_text[32:40].strip())
        except ValueError:
            print ("WARNING: There is no data of seabed_elevation")
            seabed_elevation = 0.0

        try:
            water_depth = float(line_text[40:48].strip())
        except ValueError:
            print ("WARNING: There is no data of water_depth")
            water_depth = 50.0

        water_surface_elevation = seabed_elevation + water_depth

        read_line_idxs.append(line_idx)

    return water_density, seabed_elevation, water_surface_elevation


def read_combination_infos(text_lines, read_line_idxs):
    load_combination_infos = cl.OrderedDict()

    for line_idx, line_text_0 in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not LOAD_COMBINATION_KEY == line_text_0[:len(LOAD_COMBINATION_KEY)]:
            continue

        if line_text_0[len(LOAD_COMBINATION_KEY):].strip() == '':
            # empty group line
            read_line_idxs.append(line_idx)
            continue

        load_combination_name = line_text_0[5:11].strip() # double-check this
        if load_combination_name not in load_combination_infos:
            load_combination_infos[load_combination_name] = []
        #start_col_idx = 11
        for idx in range(11, len(line_text_0), 10):
            if idx+10 > len(line_text_0):
                break
            load_label = line_text_0[idx:idx+4].strip()
            load_coefficient = convert_str_to_float_with_default(line_text_0[idx+4:idx+10])
            load_combination_infos[load_combination_name].append((load_label, load_coefficient))

        read_line_idxs.append(line_idx)

    return load_combination_infos


def read_nodes(text_lines, read_line_idxs):
    node_labels = {}
    node_coords = []
    constraint_nodes = {}
    linear_spring_nodes = {}

    #node_compress_mapping = -1 * np.ones(max_node_id)
    coords_dim = 3
    num_nodes = 0
    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not NODE_KEY == line_text[:len(NODE_KEY)]:
            continue

        items = line_text.split()

        if len(items) == 1:
            continue

        node_label = items[1].strip()

        # Joint - Spring
        if node_label in node_labels.keys():
            if not line_text[54:60] in ["ELASTI"]:
                continue
            # Translational components
            x_value = convert_str_to_float_with_default(line_text[10:17].strip()) \
                      * FORCE_SCALING / LONGITUDINAL_DIMENSION_SCALING
            y_value = convert_str_to_float_with_default(line_text[17:24].strip()) \
                      * FORCE_SCALING / LONGITUDINAL_DIMENSION_SCALING
            z_value = convert_str_to_float_with_default(line_text[24:31].strip()) \
                      * FORCE_SCALING / LONGITUDINAL_DIMENSION_SCALING
            # TODO: add rotational components
            linear_spring_nodes[node_label] = (x_value, y_value, z_value)
            if node_label in constraint_nodes:
                # constraint is now replaced by linear spring
                del constraint_nodes[node_label]
            continue

        # Ordinary Joint
        node_labels[node_label] = num_nodes
        coords = []
        for j in range(coords_dim):
            coord = convert_str_to_float_with_default(line_text[11 + 7 * j:11 + 7 * (j + 1)]) * LONGITUDINAL_DIMENSION_SCALING
            additional_coord = convert_str_to_float_with_default(line_text[11 + 7 * (j + 3):11 + 7 * (j + 4)]) * XSECT_DIMENSION_SCALING
            coords.append(coord + additional_coord)

        # Joint - Constraint
        # TODO: review constraint conditions
        constraint_str = line_text[54:60].strip()
        if constraint_str == "PILEHD":
            constraint_nodes[node_label] = (0., 0., 0., 0., 0., 0.)
        else:
            allowed = set(['0', '1'])
            if len(constraint_str) == 6 and set(constraint_str) <= allowed:
                values = [None, None, None, None, None, None]
                for value_idx in range(6):
                    if constraint_str[value_idx] == '1':
                        values[value_idx] = 0.
                constraint_nodes[node_label] = tuple(values)
        num_nodes += 1
        node_coords.append(coords)
        read_line_idxs.append(line_idx)

    node_coords = np.array(node_coords, dtype=np.float64)
    return node_labels, node_coords, constraint_nodes, linear_spring_nodes


def read_beam_elem_info(text_lines, node_labels, read_line_idxs):
    beam_elem_infos = []
    beam_offset_infos = {}
    beam_joint_labels = set()

    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not line_text[:len(MEMBER_KEY)] == MEMBER_KEY:
            continue

        if line_text.strip() == MEMBER_KEY:
            continue

        if ELEM_OFFSET_KEY in line_text:
            continue

        joint_labels = line_text[7:15]
        assert not len(joint_labels)%2, "even number"
        joint_0_label = joint_labels[:len(joint_labels)//2].strip()  # string only
        joint_1_label = joint_labels[len(joint_labels)//2:].strip()  # string only

        node_0_idx = node_labels[joint_0_label]  # id only: used for calculation
        node_1_idx = node_labels[joint_1_label]  # id only: used for calculation
        propety_label = line_text[16:19].strip()

        try:
            release_0 = line_text[22:28].strip()
        except IndexError:
            release_0 = None

        try:
            release_1 = line_text[28:34].strip()
        except IndexError:
            release_1 = None


        if len(line_text) < 19 or line_text[41:45].strip() == '':
            z_ref_joint = None
        else:
            z_ref_joint = line_text[41:45].strip()
            z_ref_joint = z_ref_joint[:len(z_ref_joint)].strip()
            z_ref_joint = node_labels[z_ref_joint] #id only

        # propety_type = line_text[19:21].strip()
        if len(line_text) < 46 or line_text[45] == ' ':
            flood_att = None
        else:
            flood_att = line_text[45].strip()
            if flood_att == 'N':
                flood_att = 0
            else:
                flood_att = 1

        read_line_idxs.append(line_idx)

        # beam offset
        rotation_angle = 0.
        if line_text[:len(ELEM_GLOBAL_OFFSET_KEY)] == ELEM_GLOBAL_OFFSET_KEY:
            rotation_angle = convert_str_to_float_with_default(line_text[36:41])
            if "MEMB2" not in text_lines[line_idx+1]:
                offset_info_text = text_lines[line_idx+1]
            else:
                offset_info_text = text_lines[line_idx+2]
                read_line_idxs.append(line_idx+2)
            read_line_idxs.append(line_idx+1)
            assert ELEM_OFFSET_KEY in offset_info_text, offset_info_text
            joint_0_global_offset = []
            joint_1_global_offset = []
            for i in range(3):
                joint_0_global_offset.append(convert_str_to_float_with_default(offset_info_text[35+i*6:35+(i+1)*6])
                                             * XSECT_DIMENSION_SCALING)
                joint_1_global_offset.append(convert_str_to_float_with_default(offset_info_text[53+i*6:53+(i+1)*6])
                                             * XSECT_DIMENSION_SCALING)
            beam_offset_infos[joint_labels] = \
                ['global', joint_0_global_offset, joint_1_global_offset]

        elif line_text[:len(ELEM_LOCAL_OFFSET_KEY)] == ELEM_LOCAL_OFFSET_KEY:
            offset_info_text = text_lines[line_idx+1]
            read_line_idxs.append(line_idx+1)
            assert ELEM_OFFSET_KEY in offset_info_text, offset_info_text
            joint_0_local_offset = []
            joint_1_local_offset = []
            # Trong: fix bug getting only offset in z direction
            for i in range(3):
                joint_0_local_offset.append(convert_str_to_float_with_default(offset_info_text[35 + i * 6:35 + (i + 1) * 6])
                                            * XSECT_DIMENSION_SCALING)
                joint_1_local_offset.append(convert_str_to_float_with_default(offset_info_text[53 + i * 6:53 + (i + 1) * 6])
                                            * XSECT_DIMENSION_SCALING)
            beam_offset_infos[joint_labels] = \
                ['local', joint_0_local_offset, joint_1_local_offset]

        beam_nodes = [node_0_idx, node_1_idx] # id only
        beam_elem_infos.append(
            (joint_labels, propety_label, beam_nodes, rotation_angle, flood_att, z_ref_joint, release_0, release_1))
        beam_joint_labels.add(joint_labels)

    return beam_elem_infos, beam_offset_infos, beam_joint_labels


def read_shell_elem_infos(text_lines, node_labels, read_line_idxs):
    print ('read_shell_elem_infos')
    shell_elems = []

    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not SHELL_KEY == line_text[:len(SHELL_KEY)]:
            continue

        if line_text.strip() == SHELL_KEY:
            continue

        if "PLATE  OFFSETS" in line_text:
            continue

        plate_label = line_text[6:10].strip()
        plate_sect_name = line_text[27:30].strip()

        joint_label_strings = line_text[11:27]
        joint_label_idxs = []
        last_joint_label = None
        for i in range(len(joint_label_strings)/LABEL_LEN):
            joint_label = joint_label_strings[i*LABEL_LEN:(i+1)*LABEL_LEN].strip()
            if len(joint_label) == 0:
                joint_label = last_joint_label
            else:
                last_joint_label = joint_label

            try:
                node_idx = node_labels[joint_label]
                joint_label_idxs.append(node_idx)
            except IndexError as e:
                print ("Error:", e)
                print (joint_label_strings)
                print (line_text)
                assert False
        read_line_idxs.append(line_idx)

        shell_elems.append(joint_label_idxs)

    shell_elems = np.array(shell_elems, dtype=np.int32)
    return shell_elems


def read_grp_ov_infos(text_lines, read_line_idxs):
    grup_ov_infos = {}
    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not GRUP_OV_KEY == line_text[:len(GRUP_OV_KEY)]:
            continue

        if line_text[len(GRUP_OV_KEY):].strip() == '':
            # empty group line
            continue

        grup_ov_name = line_text[15:18].strip()
        # Trong: add exception for None flooded/non-flooded status

        ## Get marine growth override ##
        try:
            mgr_ov = line_text[18].strip()
            # marine growth override [N, None] - This attribute has up to 6 options,
            # but in the scope of this project, we only use 2 options
        except IndexError:
            mgr_ov = ""

        if mgr_ov == '':
            mgr_ov = None
        elif mgr_ov == 'N':
            mgr_ov = 0
        else:
            assert False

        ## Get flood attribute override ##
        # Trong: add exception for None flooded override info
        try:
            flood_ov = line_text[19].strip()  # marine growth override [N,F,None]
        except IndexError:
            flood_ov = " "

        if flood_ov == ' ':
            flood_ov = None
        elif flood_ov == 'N':
            flood_ov = 0
        else:
            flood_ov = 1

        ## Get cross-section override ##
        xsec_ov = line_text[26:33].strip()
        if xsec_ov == "":
            xsec_ov = None
        else:
            xsec_ov = float(xsec_ov)

        dis_area_ov = line_text[33:40].strip()
        if dis_area_ov == "":
            dis_area_ov = None
        else:
            dis_area_ov = float(dis_area_ov)

        ## Get normal drag coef. local Y ##
        try:
            cd_y = float(line_text[52:56].strip())
        except ValueError:
            cd_y = None

        ## Get normal drag coef. local Z ##
        try:
            cd_z = float(line_text[56:60].strip())
        except ValueError:
            cd_z = None

        # cd_ov = None
        if cd_y == cd_z:
            cd_ov = cd_y
        else:
            print ("cd_ov on local Y not equal cd_ov on local Z")
            assert False

        ## Get normal mass coef. local Y ##
        try:
            cm_y = float(line_text[60:64].strip())
        except ValueError:
            cm_y = None

        ## Get normal mass coef. local Z ##
        try:
            cm_z = float(line_text[64:68].strip())
        except ValueError:
            cm_z = None

        # cm_ov = None
        if cm_y == cm_z:
            cm_ov = cm_y
        else:
            print ("cm_ov on local Y not equal cm_ov on local Z")
            assert False

        grup_info = (grup_ov_name, mgr_ov, flood_ov, xsec_ov, dis_area_ov, cd_ov, cm_ov)
        grup_ov_infos[grup_ov_name] = grup_info

        read_line_idxs.append(line_idx)

    return grup_ov_infos


def read_mem_ov_infos(text_lines, node_labels, read_line_idxs):
    beam_elem_ov_infos = []

    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not line_text[:len(MEM_OV_KEY)] == MEM_OV_KEY:
            continue

        if line_text.strip() == MEM_OV_KEY:
            continue

        joint_labels = line_text[7:15]
        assert not len(joint_labels)%2, "even number"
        joint_0_label = joint_labels[:len(joint_labels)//2].strip()
        joint_1_label = joint_labels[len(joint_labels)//2:].strip()

        node_0_idx = node_labels[joint_0_label] # id only
        node_1_idx = node_labels[joint_1_label] # id only

        if line_text[19] == '':
            flood_att = None
        else:
            flood_att = line_text[19].strip() # Only Non-flood attribute available in the scope of this project
            if flood_att == 'N':
                flood_att = 0
            else:
                flood_att = 1

        ## Get normal drag coef. local Y ##
        try:
            cd_y = float(line_text[52:56].strip())
        except ValueError:
            cd_y = None

        ## Get normal drag coef. local Z ##
        try:
            cd_z = float(line_text[56:60].strip())
        except ValueError:
            cd_z = None

        cd_ov = cd_y # Truong: temporarily fix becuz parser doesn't support multi cd value
        # if cd_y == cd_z:
        #     cd_ov = cd_y
        # else:
        #     print ("cd_ov on local Y not equal cd_ov on local Z")
        #     assert False

        ## Get normal mass coef. local Y ##
        try:
            cm_y = float(line_text[60:64].strip())
        except ValueError:
            cm_y = None

        ## Get normal mass coef. local Z ##
        try:
            cm_z = float(line_text[64:68].strip())
        except ValueError:
            cm_z = None

        cm_ov = cm_y # Truong: temporarily fix becuz parser doesn't support multi cm value
        # if cm_y == cm_z:
        #     cm_ov = cm_y
        # else:
        #     print ("cm_ov on local Y not equal cm_ov on local Z")
        #     assert False

        beam_nodes = [node_0_idx, node_1_idx] # id only
        beam_elem_ov_infos.append((joint_labels, flood_att, beam_nodes, cd_ov, cm_ov))

        read_line_idxs.append(line_idx)
    return beam_elem_ov_infos


def read_mgr_ov_infos(text_lines, read_line_idxs, seabed_elevation):
    mgr_ov_infos = []
    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not MGR_OV_KEY == line_text[:len(MGR_OV_KEY)]:
            continue

        if line_text[len(MGR_OV_KEY):].strip() == '':
            # empty group line
            continue
        # In the scope of this project, only 4 properties are extracted from MGROV
        mgr_elev_0 = float(line_text[8:16].strip())
        # Trong: temporary fix for the new definement of marine growth
        # TODO: Truong: update for LIC plugin code
        if line_text[16:24].strip() == "":
            if MGR_OV_KEY in text_lines[line_idx+1]:
                mgr_elev_1 = float(text_lines[line_idx+1][8:16].strip())
            else:
                mgr_elev_1 = mgr_elev_0
        else:
            mgr_elev_1 = float(line_text[16:24].strip())
        mgr_elev_0 = seabed_elevation + mgr_elev_0
        mgr_elev_1 = seabed_elevation + mgr_elev_1

        mgr_thickness = float(line_text[24:32].strip())
        mgr_density = float(line_text[48:56].strip())

        mgr_ov_infos.append((mgr_elev_0, mgr_elev_1, mgr_thickness, mgr_density))

        read_line_idxs.append(line_idx)

    return mgr_ov_infos


def read_cdm_infos(text_lines, read_line_idxs):
    cdm_infos = []
    for line_idx, line_text in enumerate(text_lines):
        if line_idx in read_line_idxs:
            continue

        if not CDM_KEY == line_text[:len(CDM_KEY)]:
            continue

        if line_text[len(CDM_KEY):].strip() == '':
            # empty group line
            continue
        # In the scope of this project, only 4 properties are extracted from MGROV
        diameter = float(line_text[6:12].strip())/100
        cd_normal_clean = float(line_text[12:18].strip())
        cm_normal_clean = float(line_text[24:30].strip())
        cd_normal_foul = float(line_text[36:42].strip())
        cm_normal_foul = float(line_text[48:54].strip())

        cdm_infos.append((diameter, cd_normal_clean, cm_normal_clean, cd_normal_foul, cm_normal_foul))

        read_line_idxs.append(line_idx)

    return cdm_infos


def separate_shell_elems(shell_elems):
    # Separate by connection
    G = nx.Graph()
    for shell_elem in shell_elems:
        G.add_nodes_from(shell_elem)
        G.add_edges_from([
            shell_elem[:2], shell_elem[1:3], shell_elem[2:4],
            [shell_elem[3], shell_elem[0]]])
    graphs = list(nx.connected_component_subgraphs(G))
    connection_seperated_elem_data = []
    print ("Shell connection:", len(graphs))
    if len(graphs) > 1:
        for graph_idx, graph in enumerate(graphs):
            sub_elem_data = []
            for shell_elem in shell_elems:
                test_node = shell_elem[0]
                if test_node in graph.nodes:
                    sub_elem_data.append(shell_elem)
            connection_seperated_elem_data.append(sub_elem_data)
    else:
        connection_seperated_elem_data.append(shell_elems)

    # Separate by properties (material, section, etc)
    shell_component_elems = []
    for comp_idx, elem_group in enumerate(connection_seperated_elem_data):
        block_elems = {}
        block_elem_idx = {}
        for comp_elem_idx, elem in enumerate(elem_group):
            new_key = "UNDEFINED"
            if new_key not in block_elems.keys():
                block_elems[new_key] = [elem]
                block_elem_idx[new_key] = [comp_elem_idx]
            else:
                block_elems[new_key].append(elem)
                block_elem_idx[new_key].append(comp_elem_idx)
        shell_component_elems.append(block_elems)

    return shell_component_elems

# def write_shell_component_elems_to_exo_data(shell_component_elems, node_coords, beam_elem_infos):
#     print 'write_shell_component_elems_to_exo_data'
#     #import block_attribute_tools as bt
#
#     free_beam_infos = beam_elem_infos
#     shell_exo_files = []
#     #component_exos = []
#     component_block_attributes = []
#     component_stiffener_info = []
#     #component_subdomain_mass_density = []
#
#     # akselos_material_dict = {}
#     # for mat_id in material_data:
#     #     # TODO THUC: Correct scaling
#     #     E = float(material_data[mat_id]['EX'][0])
#     #
#     #     try:
#     #         nu = float(material_data[mat_id]['NUXY'][0])
#     #     except KeyError:
#     #         nu = float(material_data[mat_id]['PRXY'][0])
#     #         material_data[mat_id]['NUXY'] = material_data[mat_id]['PRXY']
#     #
#     #     try:
#     #         rho = float(material_data[mat_id]['DENS'][0])
#     #         # convert rho from kg/mm^3 to kg/m^3
#     #         rho = 1.e9 * rho
#     #     except KeyError:
#     #         rho = None#7850.
#     #         print ("WARNING: Missing DENS in MPDATA ID", mat_id
#     #     akselos_material_dict[mat_id] = bt.ElasticityMaterial.create(
#     #         str(mat_id), E/MODULI_SCALING, nu, rho)
#
#     for comp_idx, block_elem_dict in enumerate(shell_component_elems):
#         _component_stiffener_info = {}
#
#         block_idx = 0
#         elem_start_idx = 0
#         exo_block_datas = []
#         block_attribute = {}
#         #subdomain_mass_density = {}
#         for key, block_elems in block_elem_dict.items():
#             elem_field_values = {}
#             block_elems = np.array(block_elems)[:, [0, 1, 3, 2]]
#             exo_block_data = ed.ExoBlockData(
#                 block_idx, "SHELL4", block_elems, elem_start_idx, elem_field_values)
#
#             elem_start_idx += len(block_elems)
#             exo_block_datas.append(exo_block_data)
#             block_idx += 1
#             #material_id, _, _, real_constant_id, section_id, _ = key
#             #thickness = section_types[section_id][-1]
#             #block_attribute[block_idx] = \
#             #    bt.BlockAttributes.create(akselos_material_dict[material_id], thickness)
#
#             #subdomain_mass_density[block_idx] = akselos_material_dict[material_id].mass_density
#
#         component_block_attributes.append(block_attribute)
#         #component_subdomain_mass_density.append(subdomain_mass_density)
#
#         # Check and filter beam elems
#         free_beam_infos, nodeset_list, stiffener_blocks = \
#             filter_beam_elems(free_beam_infos, exo_block_datas, node_coords)

        # stiffener_ids = []
        # stiffener_id = 1000
        # elem_start_idx = exo_block_datas[-1].elem_start_idx + len(
        #     exo_block_datas[-1].elems)
        # for key, stiffener_block in stiffener_blocks.items():
        #     block_idx = len(exo_block_datas)
        #     stiffener_block = np.array(stiffener_block, dtype=np.int64)
        #     stiffener_block_data = ed.ExoBlockData(
        #         block_idx, "BAR2", stiffener_block, elem_start_idx)
        #     elem_start_idx += len(stiffener_block)
        #     exo_block_datas.append(stiffener_block_data)
        #     stiffener_ids.append(stiffener_id)
        #     _component_stiffener_info[stiffener_id] = key
        #
        #     stiffener_id += 1
            # rho = None
            # material_props = material_data[key[0]]
            # if 'DENS' in material_props:
            #     rho = float(material_data[key[0]]['DENS'][0])
            #     # convert rho from kg/mm^3 to kg/m^3
            #     rho = 1.e9 * rho
            # component_subdomain_mass_density[stiffener_id] = rho

        # nodeset_id = 2000
        # nodesets = {}
        # for coord_idx in nodeset_list:
        #     nodesets[nodeset_id] = np.array([coord_idx])
        #     nodeset_id += 1
        #
        # block_ids = np.arange(len(exo_block_datas)) + 1
        # sideset_elem_idxs = {}
        # exo_fside_idxs = {}
        # exo_data = ed.ExoData(
        #     node_coords, exo_block_datas, sideset_elem_idxs, exo_fside_idxs, nodesets,  block_ids)

        # f = tempfile.NamedTemporaryFile(
        #     delete=False, prefix="component"+str(comp_idx)+"_", suffix=".exo")
        # print (" > Writing exo data (shell) to", f.name

        #f.close()
        #reordered_exo_data = et.renumber_element_nodes(exo_data)
        #ip.write_inp_from_exo_data(reordered_exo_data,
        # 'F:\COLS\collections\\thucnguyen\Test2\shell.inp')
        #exo_data.write(f.name, format="NETCDF4")
        #shell_exo_files.append(f.name)
        #component_stiffener_info.append(_component_stiffener_info)

    # return shell_exo_files, free_beam_infos, component_stiffener_info
#




def filter_beam_elems(beam_infos, component_coords, global_coords, all_shell_elems, progress_bar):
    free_beam_infos = []
    tKDT = sp.KDTree(component_coords)
    for beam_info in beam_infos:
        joint_labels, property_label, beam_elem, rotation_angle, _, _, _, _ = beam_info
        beam_coords = global_coords[beam_elem]
        distances, node_idxs = tKDT.query(beam_coords)
        if max(distances) <= 1e-5:
            # It means both two beam nodes belong to shell component nodes
            first_node_idx = node_idxs[0]
            edge_vectors, _ = get_all_edge_vectors_from_node(
                first_node_idx, all_shell_elems, component_coords)  # loop through all shell comps
            beam_vector = beam_coords[1] - beam_coords[0]
            unit_vector = beam_vector/np.linalg.norm(beam_vector)
            test = np.dot(edge_vectors, unit_vector)
            if np.any(test >= 1-1e-4):
                progress_bar.increase()
                # it should be stiffener, not free beam
                continue

        free_beam_infos.append(beam_info)
        progress_bar.increase()
    progress_bar.exit()
    print("  -> {} beams are detected as stiffener, {} free beams left"\
        .format(len(beam_infos)-len(free_beam_infos), len(free_beam_infos)))

    return free_beam_infos


def get_regular_tube(dimensions):
    dimensions = np.array(dimensions) * 100.0  # Trong: value in name is in cm unit
    sect_type = "STUB"
    first_dim = ('%.6s' % dimensions[0])
    if len(first_dim) == 5:
        first_dim += '0'
    if len(first_dim) == 4:
        first_dim += '00'
    mesh_beam_name = first_dim + '_' + ('%.3f' % dimensions[1]) + \
        '_' + sect_type + ".exo"
    return mesh_beam_name


def merge_beam_elems(G, coords):
    found = True

    nodes_2 = set([])
    for node in G.nodes:
        if len(list(G.adj[node])) == 2:
            nodes_2.add(node)

    while found:
        found = False
        keys = nx.get_edge_attributes(G, 'key')
        for node in nodes_2:
            new_edge = list(G.adj[node])
            if len(new_edge) != 2:
                continue
            #print ("node", node

            edges_at_node = list(G.edges(node))
            _edge_0 = edges_at_node[0]
            _edge_1 = edges_at_node[1]

            if _edge_0 in keys:
                key_0 = keys[_edge_0]
            elif _edge_0[::-1] in keys:
                key_0 = keys[_edge_0[::-1]]
            else:
                assert False

            if _edge_1 in keys:
                key_1 = keys[_edge_1]
            elif _edge_1[::-1] in keys:
                key_1 = keys[_edge_1[::-1]]
            else:
                assert False

            if key_0 != key_1:
                continue

            if new_edge in G.edges or new_edge[::-1] in G.edges:
                continue

            if _edge_0[0] == node:
                _edge_0 = _edge_0[::-1]
            if _edge_1[1] == node:
                _edge_1 = _edge_1[::-1]

            v0 = coords[_edge_0[1], :] - coords[_edge_0[0], :]
            v1 = coords[_edge_1[1], :] - coords[_edge_1[0], :]
            v0 = v0/np.linalg.norm(v0)
            v1 = v1/np.linalg.norm(v1)

            if np.dot(v0, v1) < 0.90:
                continue

            for edge in edges_at_node:
                G.remove_edge(*edge)

            G.add_edges_from([new_edge], key=key_0)
            found = True
            break

    beam_keys = nx.get_edge_attributes(G, 'key')
    return G.edges, beam_keys


def get_predefined_beam_type(sect_type, dimensions):
    dim_dict = {}
    if sect_type == "":
        # tubular cross-section, existing in Akselos predefined beams
        dim_dict = {
            "radius": dimensions[0]/2, "thickness": dimensions[1]
        }
        return 'predefined_beams/circular_tube', dim_dict
    elif sect_type[:3] == "CHL":
        # c_section, existing in Akselos predefined beams
        dim_dict = {
            "height": dimensions[0], "width": dimensions[1],
            "height_thickness": dimensions[3], "width_thickness": dimensions[2]
        }
        return 'predefined_beams/c_section', dim_dict
    elif sect_type[:3] in ["WF", "PLG"]:
        dim_dict = {
            "H": dimensions[2]-2.0*dimensions[1], "B": dimensions[0],
            "h": dimensions[1], "b": dimensions[3]
        }
        # i_section, existing in Akselos predefined beams
        return 'predefined_beams/i_section', dim_dict
    elif sect_type[:3] == "PRI":
        # rect_section, existing in Akselos predefined beams
        dim_dict = {
            "height": dimensions[0], "width": dimensions[2]
        }
        return 'predefined_beams/rectangular', dim_dict
    elif sect_type[:3] == "BOX":
        # rect_tube_section, existing in Akselos predefined beams
        dim_dict = {
            "height": dimensions[0], "width": dimensions[2],
            "height_thickness": dimensions[1], "width_thickness": dimensions[3]
        }
        return 'predefined_beams/rectangular_tube', dim_dict
    elif sect_type[:3] == "ANG":
        dim_dict = {
            "height": dimensions[0], "width": dimensions[1],
            "wall_thickness": dimensions[2]
        }
        return 'predefined_beams/l_section', dim_dict
    elif sect_type[:3] == "TUB":
        # tubular cross-section, existing in Akselos predefined beams
        # Trong: considering the case of double tub in SACS by the third dimensions
        if dimensions[2] > 1.0e-6:
            return None, dim_dict
        else:
            dim_dict = {
                "radius": dimensions[0]/2, "thickness": dimensions[1]
            }
        return 'predefined_beams/circular_tube', dim_dict
    elif sect_type[:3] == "PGB":
        return None, dim_dict
    # elif sect_type[:3] == "CON":
    #     return None, dim_dict
    else:
        return None, dim_dict


def set_predefined_beam_geometry(beam, beam_type_name, dime_dict):
    # Have to convert from cm to m
    for parameter in dime_dict:
        value = dime_dict[parameter]
        if value < beam.mu_geo[parameter].min:
            print (beam_type_name, parameter, "min", beam.mu_geo[parameter].min, value)
            assert False
        elif value > beam.mu_geo[parameter].max:
            print (beam_type_name, parameter, "max", beam.mu_geo[parameter].max, value)
            assert False
        beam.mu_geo[parameter].value = value

    if not beam.is_predefined_beam():
        # TODO: redundant
        return

    beam.update_cross_section_properties()


def convert_str_to_float_with_default(text):
    try:
        v = float(text)
    except ValueError:
        v = 0.

    return v


def find_beam_transformation(target_nodes, z_ref_joint_coord=None, is_spring=False):
    def normalize(v):
        return v / np.linalg.norm(v)

    beam_axis = target_nodes[1]-target_nodes[0]
    beam_length = np.linalg.norm(beam_axis)

    mapped_nodes = beam_length*BEAM_DEFAULT_NODES
    result = mt.find_best_transformation(mapped_nodes, target_nodes)
    #beam.rotation_matrix = result[:3, :3]
    translation_vector = result[:3, 3]

    # Correct t1 and t2 beam orientation
    beam_axis = normalize(beam_axis)
    tolerance = 1.0-1e-10

    if z_ref_joint_coord is not None and not is_spring:
        ref_vector = z_ref_joint_coord - target_nodes[0]
        xz_plane_normal = normalize(np.cross(beam_axis, ref_vector)) #xz_plane of local coord.
        t2_direction = normalize(np.cross(beam_axis, xz_plane_normal))
        if np.dot(t2_direction, ref_vector) < 0.:
            t2_direction = -1.0 * t2_direction
        t1_direction = normalize(np.cross(t2_direction, beam_axis))
    else:
        if np.abs(np.dot(beam_axis, GLOBAL_Z)) < tolerance:
            xZ_plane_normal = normalize(np.cross(beam_axis, GLOBAL_Z))
            t2_direction = normalize(np.cross(beam_axis, xZ_plane_normal))
            if np.dot(t2_direction, GLOBAL_Z) < 0.:
                t2_direction = -1.0*t2_direction
            t1_direction = normalize(np.cross(t2_direction, beam_axis))
        else:
            if np.dot(beam_axis, GLOBAL_Z) >= tolerance:
                beam_axis = GLOBAL_Z
                t1_direction = GLOBAL_X
                t2_direction = GLOBAL_Y
            else:
                beam_axis = -GLOBAL_Z
                t1_direction = -GLOBAL_X
                t2_direction = GLOBAL_Y
    rotation_matrix = np.eye(3)
    rotation_matrix[:, 0] = beam_axis
    rotation_matrix[:, 1] = t1_direction
    rotation_matrix[:, 2] = t2_direction

    # CHECKING
    #result[:3, :3] = beam.rotation_matrix
    #print mesh.mesh_port_tools.transform_array(result, c)

    return rotation_matrix, translation_vector, beam_length


def rotate_beam(rotation_angle, rotation_matrix):
    a = np.deg2rad(rotation_angle)
    u = rotation_matrix[:, 0]
    cos = np.cos(a)
    sin = np.sin(a)
    r00 = cos+(1-cos)*u[0]**2
    r01 = u[0]*u[1]*(1-cos)-u[2]*sin
    r02 = u[0]*u[2]*(1-cos)+u[1]*sin
    r10 = u[1]*u[0]*(1-cos)+u[2]*sin
    r11 = cos+(1-cos)*u[1]**2
    r12 = u[1]*u[2]*(1-cos)-u[0]*sin
    r20 = u[2]*u[0]*(1-cos)-u[1]*sin
    r21 = u[2]*u[1]*(1-cos)+u[0]*sin
    r22 = cos+(1-cos)*u[2]**2
    rot = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]])
    return np.dot(rot, rotation_matrix)


def get_spring_coords(beam_coords, port_idx, spring_direction):
    assert port_idx in [0, 1]
    spring_root_node = beam_coords[port_idx]
    beam_axis = beam_coords[-1] - beam_coords[0]
    # beam_axis = beam_axis/np.linalg.norm(beam_axis)
    # global_z = np.array([0., 0., 1.])
    # if np.dot(beam_axis, global_z) > 1-1e-8:
    #     # align with gloabl z
    other_node = spring_root_node + np.array(spring_direction)*np.linalg.norm(beam_axis)/10.
    return np.array([spring_root_node, other_node])


def read_json_file(filepath):
    with open(filepath, "r") as f_file:
        content = f_file.read()
        f_file.close()
    return json.loads(content)


def load_external_defined_xsecs(external_xsec_lib_filepath=None):
    if external_xsec_lib_filepath is None:
        external_xsec_lib_filepath = os.path.join(directories.TOOLS_DIR, 'akselos/mesh/mesh_template/lib_xsec/aks_xsec.json')
    external_xsec_lib = read_json_file(external_xsec_lib_filepath)
    return external_xsec_lib


def get_external_defined_xsecs(sec_name, sect_lib, return_stiffener_mesh=False):
    standard = AKSELOS_DEFAULT_XSECT_STANDARD
    if sec_name not in sect_lib[standard].keys():
        print("Warn: default standard %s does not support this cross-section %s" % (standard, sec_name))
        check = False
        for std in sect_lib.keys():
            if sec_name in sect_lib[std].keys():
                check = True
                standard = std
                print("Use standard %s instead" % (standard))
                break
        assert check, "Does not support this cross-section %s" % (sec_name)
    #
    unit = sect_lib[standard][sec_name]['unit']
    external_xsect_type = sect_lib[standard][sec_name]['type']
    scaling = convert_units('length', '{} to {}'.format(unit, AKSELOS_LENGTH_UNIT))
    dimension_values = list(scaling * np.array(sect_lib[standard][sec_name]['params']))
    #
    if return_stiffener_mesh:
        return sec_name + '_' + external_xsect_type + ".exo", None #TODO: review
    map_xsect_type_from_external_lib_to_akselos = {'ANG': 'l', 'L': 'l',
                                        'W': 'i', 'IPE': 'i', 'HE': 'i', 'WF': 'i', 'PLG': 'i',
                                        'MC': 'c', 'C': 'c', 'CHL': 'c',
                                        'PRI': 'rectangular',
                                        'BOX': 'rectangular_tube'}
    assert external_xsect_type in map_xsect_type_from_external_lib_to_akselos.keys(), "Please update the map"
    map_dimension_from_external_lib_to_akselos = {'ANG': ['H', 'W', 't'],
                     'PGB': ['W', 'th', 'H', 'tw'],
                     'WF': ['W', 'th', 'H', 'tw'],
                     'PLG': ['W', 'th', 'H', 'tw'],
                     'TUB': ['do', 'to', 'di', 'ti'],
                     'W': ['H', 'W', 'tw', 'th'],
                     'MC': ['H', 'W', 'th', 'tw'],
                     'L': ['H', 'W', 't'],
                     'IPE': ['H', 'W', 'tw', 'th'],
                     'HE': ['H', 'W', 'tw', 'th'],
                     'C': ['H', 'W', 'th', 'tw'],
                     'CHL': ['H', 'W', 'tw', 'th'],
                     'BOX': ['H', 'tw', 'W', 'th'],
                     'PRI': ['H', 'tmp', 'W'],
                     'STUB': ['d', 't']}
    geo_params = {}
    for i, key in enumerate(map_dimension_from_external_lib_to_akselos[external_xsect_type], start=0):
        geo_params[key] = dimension_values[i]
    akselos_xsect_type = map_xsect_type_from_external_lib_to_akselos[external_xsect_type]
    dim_dict = {}
    if akselos_xsect_type == 'i':
        dim_dict = {
            "H": geo_params['H'] - 2.0*geo_params['th'], "B": geo_params['W'],
            "h": geo_params['th'], "b": geo_params['tw']
        }
        akselos_xsect_type = 'predefined_beams/i_section'
        # TODO: review .. - 2.0*geo_params['th']
    elif akselos_xsect_type == 'c':
        dim_dict = {
            "height": geo_params['H'], "width": geo_params['W'],
            "height_thickness": geo_params['th'], "width_thickness": geo_params['tw']
        }
        akselos_xsect_type = 'predefined_beams/c_section'
    elif akselos_xsect_type == 'l':
        dim_dict = {
            "height": geo_params['H'], "width": geo_params['W'],
            "wall_thickness": geo_params['t']
        }
        akselos_xsect_type = 'predefined_beams/l_section'
    elif akselos_xsect_type == 'rectangular':
        dim_dict = {
            "height": geo_params['H'], "width": geo_params['W']
        }
        akselos_xsect_type = 'predefined_beams/rectangular'
    elif akselos_xsect_type == 'rectangular_tube':
        dim_dict = {
            "height": geo_params['H'], "width": geo_params['W'],
            "height_thickness": geo_params['th'], "width_thickness": geo_params['tw']
        }
        akselos_xsect_type = 'predefined_beams/rectangular_tube'

    # TODO: standardize all x-section name as well as dimensions in component_type.json, e.g rectangular -> rect_, ...
    return akselos_xsect_type, dim_dict


def is_node_inside_shell_area(shell_nodes, coords, middle_node_coord):
    def sign_p(p0, p1, p2):
        return (p0[0]-p2[0])*(p1[1]-p2[1])-(p1[0]-p2[0])*(p0[1]-p2[1])

    def point_in_triangle(pt, v1, v2, v3):
        b1 = sign_p(pt, v1, v2) <= 0.0
        b2 = sign_p(pt, v2, v3) <= 0.0
        b3 = sign_p(pt, v3, v1) <= 0.0

        return b1 == b2 and b2 == b3

    for elem in shell_nodes:
        nodes = coords[elem]
        nodes = np.concatenate((nodes, [middle_node_coord]))
        t1 = (nodes[0] - nodes[1])
        t1 = t1 / np.linalg.norm(t1)
        n = np.cross(nodes[0] - nodes[1], nodes[1] - nodes[2])
        n = n / np.linalg.norm(n)
        t2 = np.cross(n, t1)
        nodes_2d = np.dot(nodes, np.array([t1, t2]).transpose())
        #middle_pt_2d = np.dot([middle_node_coord], np.array([t1, t2]).transpose())[0]

        in_triangle_0 = point_in_triangle(nodes_2d[-1], nodes_2d[0], nodes_2d[1], nodes_2d[2])
        in_triangle_1 = point_in_triangle(nodes_2d[-1], nodes_2d[2], nodes_2d[3], nodes_2d[0])

        if in_triangle_0 or in_triangle_1:
            return True
    return False


def split_beams(free_beam_infos, split_beam_infos, node_coords):
    new_beam_infos = set()
    splitted_beam_info = {}

    for idx, beam_info in enumerate(free_beam_infos):
        joint_labels, prop_id, beam_elem, rotation_angle = beam_info
        beam_coords = node_coords[beam_elem, :]
        node_0 = beam_coords[0]
        node_1 = beam_coords[1]
        if joint_labels not in split_beam_infos:
            new_beam_infos.add((joint_labels, prop_id, tuple(node_0), tuple(node_1)))
            continue

        splitted_beam_info[joint_labels] = []

        beam_axis = beam_coords[1] - beam_coords[0]
        beam_length = np.linalg.norm(beam_axis)
        beam_axis = beam_axis/beam_length
        split_from_node_a_info, split_from_node_b_info = split_beam_infos[joint_labels]
        split_from_node_a_info_0 = list(split_from_node_a_info)
        split_from_node_a_info_1 = []
        is_valid = True
        for right_offset in split_from_node_b_info:
            if right_offset > beam_length:
                print ("ERROR")

                is_valid = False
                # assert False
            split_from_node_a_info_1.append(beam_length-right_offset)

        if is_valid:
            new_beam_infos.add((joint_labels, prop_id, tuple(node_0), tuple(node_1)))
            continue
        split_positions = sorted(split_from_node_a_info_0+split_from_node_a_info_1)

        assert split_positions[0] > 0., split_positions
        split_positions.insert(0, 0.)
        assert split_positions[-1] < beam_length, split_positions
        split_positions.append(beam_length)
        for idx,  split_position in enumerate(split_positions):
            if idx == len(split_positions) - 1:
                continue
            start_node = node_0 + split_positions[idx] * beam_axis
            end_node = node_0 + split_positions[idx+1] * beam_axis
            #print ("add", start_node, end_node)
            sub_beam_label = joint_labels + "-" + str(idx)
            new_beam_infos.add((joint_labels, prop_id, tuple(start_node), tuple(end_node)))
            splitted_beam_info[joint_labels].append(sub_beam_label)

    return new_beam_infos, splitted_beam_info


def calculate_stiffener_orientations(beam_elem, coords):

    beam_axis = coords[beam_elem[1]] - coords[beam_elem[0]]
    beam_length = np.linalg.norm(beam_axis)
    beam_axis = beam_axis / beam_length

    if np.abs(np.dot(beam_axis, GLOBAL_Z)) < 1.0-1e-10:
        xZ_plane_normal = np.cross(beam_axis, GLOBAL_Z)
        t2_direction = np.cross(beam_axis, xZ_plane_normal)
        if np.dot(t2_direction, GLOBAL_Z) < 0.:
            t2_direction = -1.0*t2_direction
        t1_direction = np.cross(t2_direction, beam_axis)
        t1_direction = t1_direction/np.linalg.norm(t1_direction)
        t2_direction = t2_direction/np.linalg.norm(t2_direction)
    else:
        beam_axis = GLOBAL_Z
        t1_direction = GLOBAL_X
        t2_direction = GLOBAL_Y

    akselos_t2_direction = -t1_direction
    akselos_t1_direction = t2_direction

    return beam_axis, akselos_t1_direction, akselos_t2_direction


def convert_to_float(item):
    item0 = item
    item = str(item)
    f_item = None
    try:
        f_item = float(item)
    except ValueError:
        if item[0] in ['-', '+']:
            if "-" in item[1:]:
                item = item[0] + item[1:].replace('-', "E-")
            elif "+" in item[1:]:
                item = item[0] + item[1:].replace('+', "E+")
        else:
            if "-" in item:
                item = item.replace('-', "E-")
            elif "+" in item:
                item = item.replace('+', "E+")

        try:
            f_item = float(item)
        except ValueError:
            pass

    if f_item is None:
        print (" Error: cannot convert to float:", item, item0)
    return f_item


def get_all_edge_vectors_from_node(checking_node_idx, elems, coords):
    found_idxs = np.where(elems == checking_node_idx)
    u = np.unique(elems[found_idxs[0]].flatten())
    neighbor_node_idxs = np.delete(u, np.where(u==checking_node_idx)[0][0])
    origin = coords[checking_node_idx]
    edge_vectors = [np.array(coords[n]-origin) for n in neighbor_node_idxs]
    norms = np.linalg.norm(edge_vectors, axis=1)
    edge_vectors = edge_vectors*(1/np.tile(norms, (3, 1)).transpose())
    return edge_vectors, neighbor_node_idxs


def check_invalid_connector(component_system):
    import id_objects.port_id

    if IS_USE_RIGID:
        connector_name = "/builtin/rigid_beam"
    else:
        assert False

    invalid_connector_ids = set()
    components = component_system.components
    for component in component_system.components:
        if not component.is_six_dofs_rigid_connector() and not component.is_six_dofs_spring_connector():
            continue
        if component._id in invalid_connector_ids:
            continue
        is_valid = True
        neighbor_comp_infos = [(), ()]

        for port_idx in range(2):
            if not is_valid:
                continue
            _port_id = id_objects.port_id.PortId(component, port_idx, 0)
            port_connection = component.component_neighbors[_port_id]
            n_neighbors = len(port_connection.port_ids) - 1
            assert n_neighbors >= 0
            is_connected = n_neighbors > 0
            if not is_connected:
                is_valid = False
            else:
                for port_id in port_connection.port_ids:
                    if port_id.component._id == component._id:
                        continue
                    neighbor_comp_infos[port_idx] += \
                        ((port_id.component._id, port_id.port_idx),)

        if not is_valid:
            invalid_connector_ids.add(component._id)
            continue

        for port_idx in range(2):
            neighbor_infos = neighbor_comp_infos[port_idx]
            for neighbor_info in neighbor_infos:
                neighbor_comp_id, neighbor_port_idx = neighbor_info
                neighbor_comp = components[neighbor_comp_id]
                if neighbor_comp._id in invalid_connector_ids:
                    continue
                if not neighbor_comp.type.core.name == connector_name:
                    continue
                other_port_idx = 1 - neighbor_port_idx
                _port_id = id_objects.port_id.PortId(neighbor_comp, other_port_idx, 0)
                port_connection = neighbor_comp.component_neighbors[_port_id]
                for _neighbor_port_id in port_connection.port_ids:
                    if _neighbor_port_id.component._id == component._id:
                        invalid_connector_ids.add(neighbor_comp._id)
                        break

    return invalid_connector_ids


def correct_t1_t2_based_on_beam_type(beam_type_name, rotation_matrix):
    if beam_type_name in AKSELOS_TO_SACS_T1_T2_GEOMETRY_ANGLE:
        theta = AKSELOS_TO_SACS_T1_T2_GEOMETRY_ANGLE[beam_type_name]
        a = np.deg2rad(theta)
        u = rotation_matrix[:, 0]
        sina = np.sin(a)
        cosa = np.cos(a)
        ux = u[0]
        uy = u[1]
        uz = u[2]
        R = np.array([[cosa+ux**2*(1-cosa), ux*uy*(1-cosa)-uz*sina, ux*uz*(1-cosa)+uy*sina],
                      [uy*ux*(1-cosa)+uz*sina, cosa+uy**2*(1-cosa), uy*uz*(1-cosa)-ux*sina],
                      [uz*ux*(1-cosa)-uy*sina, uz*uy*(1-cosa)+ux*sina, cosa+uz**2*(1-cosa)]])
        rotation_matrix = np.dot(R, rotation_matrix)
    return rotation_matrix


def transform_local_force_on_beam(beam_type_name, rotation_matrix, local_force, debug=False):
    if beam_type_name in AKSELOS_TO_SACS_T1_T2_GEOMETRY_ANGLE:
        glocal_force = np.multiply(rotation_matrix.T, local_force[:, np.newaxis])
        glocal_force = np.sum(glocal_force, axis=0)
        # We need to rotate the force -theta because our axis were rotated theta (to make the
        # geometry correct)
        theta = -AKSELOS_TO_SACS_T1_T2_GEOMETRY_ANGLE[beam_type_name]
        a = np.deg2rad(theta)
        u = rotation_matrix[:, 0]
        sina = np.sin(a)
        cosa = np.cos(a)
        ux = u[0]
        uy = u[1]
        uz = u[2]
        Ra = np.array([[cosa+ux**2*(1-cosa), ux*uy*(1-cosa)-uz*sina, ux*uz*(1-cosa)+uy*sina],
                       [uy*ux*(1-cosa)+uz*sina, cosa+uy**2*(1-cosa), uy*uz*(1-cosa)-ux*sina],
                       [uz*ux*(1-cosa)-uy*sina, uz*uy*(1-cosa)+ux*sina, cosa+uz**2*(1-cosa)]])
        rotated_global_force = np.dot(Ra, glocal_force.T).T
        # Rotation matrix is orthogonal matrix (i.e. whose columns and rows are orthogonal unit
        # vectors). Hence, its inverse matrix is its transpose matrix.
        rotation_matrix_inv = rotation_matrix.T
        rotated_local_force = np.dot(rotation_matrix_inv, rotated_global_force).T
        return rotated_local_force
    return local_force


def create_beam(component_id, beam_coords, beam_elem, joint_labels, sect_type, dimensions,
                sect_label, rotation_angle, collection, young_modulus, mass_density,
                cross_section_mesh_path, external_sect_lib, beam_label_to_comp_id, node_labels,
                constraint_nodes, linear_spring_nodes, linear_springs, flood_att, poisson_ratio,
                z_ref_joint_coord, sub_beam_idx=None):

    # print 'BEAM ID {}, PROP: {}, SECT TYPE: {}'.format(component_id, prop_id, sect_type)
    beam_type_name, dime_dict = get_predefined_beam_type(sect_type, dimensions)
    if beam_type_name is not None:  # is predefined beam
        beam_type = collection.rb_ref_components[beam_type_name]
    else:
        mesh_beam_name = sect_label + '_' + sect_type
        mesh_beam_filepath = os.path.join(cross_section_mesh_path, mesh_beam_name)
        mesh_beam_name = 'cross_sections/' + mesh_beam_name
        if mesh_beam_name in collection.rb_ref_components:  # is mesh beam
            # print (" + Re-used beam type", cross_section_mesh_name
            beam_type = collection.rb_ref_components[mesh_beam_name]
        else:
            mesh_beam_filepath += '.exo'
            if os.path.exists(mesh_beam_filepath):
                beam_type = new_type.new_cross_section_type(mesh_beam_filepath, collection)
                ap.write_type_to_json(beam_type, directories.DATA_DIR)
            else:
                if sect_type == 'CON':
                    print (joint_labels)
                    # TODO THUC: handle CON
                    assert False
                # print ("Error: missing cross-section mesh", mesh_beam_name
                # print sect_type, mesh_beam_name
                # print sect_type, external_sect_lib
                beam_type_name, dime_dict = get_external_defined_xsecs(sect_type, external_sect_lib)
                if beam_type_name is not None:
                    beam_type = collection.rb_ref_components[beam_type_name]
                else:
                    beam_type = collection.rb_ref_components['/builtin/generic_beam']
                # beam_type = collection.rb_ref_components['/builtin/generic_beam']

    beam = rb.RBComponent(component_id, beam_type)

    rotation_matrix, translation_vector, beam_length = find_beam_transformation(
        beam_coords, z_ref_joint_coord)
    if beam.is_predefined_beam():
        # Because in our predefined beams, t1/t2 orientation are different from SACS (the
        # difference is specified in AKSELOS_TO_SACS_T1_T2_GEOMETRY_ANGLE), so here we have to
        # rotate our predefined beams consistently.
        rotation_matrix = correct_t1_t2_based_on_beam_type(beam_type_name, rotation_matrix)

    beam.mu_geo["length"].value = beam_length
    if rotation_angle > 1e-1:  # degree
        print (" Rotate beam", joint_labels, rotation_angle)
        rotation_matrix = rotate_beam(rotation_angle, rotation_matrix)
    beam.rotation_matrix = rotation_matrix
    beam.translation_vector = translation_vector

    if beam.is_predefined_beam():
        set_predefined_beam_geometry(beam, beam_type_name, dime_dict)
    beam.post_init()
    beam_tag = joint_labels[:4].strip() + '-' + joint_labels[4:].strip()
    if sub_beam_idx is not None:
        beam_tag = beam_tag + "#" + str(sub_beam_idx)

    if flood_att is 0:
        flood_att = 'unflooded'
    else:
        flood_att = 'flooded'

    beam.set_tags(set([beam_tag, flood_att]))
    if sub_beam_idx is None:
        beam_label_to_comp_id[joint_labels] = (component_id, beam_length)
    component_id += 1

    # Set beam young modulus, possion ratio and mass density
    if not beam.is_generic_beam():
        young_modulus_GPa = young_modulus
        young_modulus_value_exists = False
        poisson_ratio_value_exists = False
        assert beam_type.core.parameters[0].type == "young_modulus"
        assert beam_type.core.parameters[1].type == "poisson_ratio"
        young_modulus_parameter = beam_type.core.parameters[0]
        poisson_ratio_parameter = beam_type.core.parameters[1]

        for discrete_value in young_modulus_parameter.discrete_values:
            if discrete_value.value == young_modulus_GPa:
                young_modulus_value_exists = True
                break

        for discrete_value in poisson_ratio_parameter.discrete_values:
            if discrete_value.value == poisson_ratio:
                poisson_ratio_value_exists = True
                break

        should_write_young_modulus = False
        if not young_modulus_value_exists:
            nondimensional_E = dv.decode_param(
                collection, "", young_modulus_GPa, type="young_modulus")
            new_parameter_E = tp.DiscreteParameterValue(
                name=str(young_modulus_GPa), value=nondimensional_E)
            young_modulus_parameter.discrete_values.append(new_parameter_E)
            should_write_young_modulus = True

        should_write_poisson_ratio = False
        if not poisson_ratio_value_exists:
            nondimensional_v = dv.decode_param(
            collection, "", poisson_ratio, type="poisson_ratio")
            new_parameter_v = tp.DiscreteParameterValue(
            name = str(poisson_ratio), value = nondimensional_v)
            poisson_ratio_parameter.discrete_values.append(new_parameter_v)
            should_write_poisson_ratio = True

        if should_write_young_modulus or should_write_poisson_ratio:
            ap.write_type_to_json(beam_type, directories.DATA_DIR)

        mass_density_modified = modify_density_double_tub(mass_density, sect_type, dimensions)

        nondimensional_mass_density = dv.decode_param(
            collection, "", mass_density_modified, type="mass_density")
        beam.mu_coeff['E_1'].value = nondimensional_E
        beam.mu_coeff['nu_1'].value = poisson_ratio
        assert 'density_1' in beam.mu_coeff
        beam.mu_coeff['density_1'].value = nondimensional_mass_density

    # Constraint beam node
    x = id_objects.port_id.PortConstraintData("x", "displacement")
    y = id_objects.port_id.PortConstraintData("y", "displacement")
    z = id_objects.port_id.PortConstraintData("z", "displacement")
    theta_x = id_objects.port_id.PortConstraintData("theta_x", None)
    theta_y = id_objects.port_id.PortConstraintData("theta_y", None)
    theta_z = id_objects.port_id.PortConstraintData("theta_z", None)
    group_0 = id_objects.port_id.PortConstraintGroup(
        "(x,y,z)", [x, y, z, theta_x, theta_y, theta_z])
    groups = [group_0]
    builtin_spring = collection.rb_ref_components['/builtin/builtin_spring']
    n_linear_springs = 0
    for node_idx in range(2):
        joint = joint_labels[4 * node_idx:4 * (node_idx + 1)].strip()
        mesh_node_idx = node_labels[joint]
        port_idx = beam_elem.index(mesh_node_idx)
        if joint in constraint_nodes:
            values = constraint_nodes[joint]
            _port_id = id_objects.port_id.PortId(beam, port_idx, 0)
            port_connection = _port_id.component.component_neighbors[_port_id]
            port_connection.constraint_parameters.port_constraint_groups = groups
            for dir_idx, direction in enumerate(['x', 'y', 'z', 'theta_x', 'theta_y', 'theta_z']):
                if values[dir_idx] is not None:
                    port_connection.constraint_parameters.set_constraint(direction, values[dir_idx])

        if joint in linear_spring_nodes:
            values = linear_spring_nodes[joint]
            for dir_idx, spring_direction in enumerate([(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]):
                if values[dir_idx] < 1e-8:
                    continue
                linear_spring = rb.RBComponent(component_id, builtin_spring)
                n_linear_springs += 1
                spring_coords = get_spring_coords(beam_coords, port_idx, spring_direction)
                rotation_matrix, translation_vector, spring_len = \
                    find_beam_transformation(spring_coords, is_spring=True)

                linear_spring.mu_geo["length"].value = spring_len
                linear_spring.mu_coeff["spring_constant"].value = values[dir_idx]
                linear_spring.rotation_matrix = rotation_matrix
                linear_spring.translation_vector = translation_vector

                linear_spring.boolean_data["specify_direction"] = True
                linear_spring.mu_coeff["direction_x"].value = spring_direction[0]
                linear_spring.mu_coeff["direction_y"].value = spring_direction[1]
                linear_spring.mu_coeff["direction_z"].value = spring_direction[2]
                linear_spring.post_init()
                linear_springs.append(linear_spring)
                component_id += 1

                # clamp second node
                _port_id = id_objects.port_id.PortId(linear_spring, 1, 0)
                port_connection = _port_id.component.component_neighbors[_port_id]
                port_connection.constraint_parameters.port_constraint_groups = groups
                for dir_idx, direction in enumerate(['x', 'y', 'z', 'theta_x', 'theta_y', 'theta_z']):
                    port_connection.constraint_parameters.set_constraint(direction, 0.)

    return beam, n_linear_springs


def modify_density_double_tub(mass_density, sect_type, dimensions):
    modified_density = mass_density
    if sect_type[:3] == "TUB":  # Recheck for both models: Malay_Jacket and Dulang to cofirm no bug here
        if not dimensions[2] > 1.0e-6:
            rho_concrete = 2400  # kg/m3
            radius_out = dimensions[0]/2
            thickness_out = dimensions[1]
            radius_in = dimensions[2]/2
            thickness_in = dimensions[3]

            area_outside_tub = np.pi*np.square(radius_out) - np.pi*np.square(radius_out - thickness_out)
            area_inside_tub = np.pi*np.square(radius_in) - np.pi*np.square(radius_in - thickness_in)
            area_tub = area_outside_tub + area_inside_tub
            area_concrete = np.pi*np.square(radius_out - thickness_out) - np.pi*np.square(radius_in)

            modified_density = (area_tub*mass_density + area_concrete*rho_concrete)/area_tub

    return modified_density


def calculate_sub_beam_concentrated_load_position(load_position, sub_beam_position):
    if sub_beam_position[1] < load_position or load_position < sub_beam_position[0]:
        return None
    elif load_position == sub_beam_position[0]:
        return 0.0
    elif load_position == sub_beam_position[1]:
        return sub_beam_position[1] - sub_beam_position[0]
    else:
        return load_position - sub_beam_position[0]
    return subbeam_offset


def calculate_sub_beam_linear_load(start_load_vector, end_load_vector,
                                   load_position, sub_beam_position):
    load_diff_vector = end_load_vector-start_load_vector
    load_length = load_position[1] - load_position[0]
    if sub_beam_position[0] < load_position[0]:
        if sub_beam_position[1] <= load_position[0]:
            return None, None, None, None
        elif load_position[0] <= sub_beam_position[1] < load_position[1]:
            sub_beam_start_load_position = load_position[0] - sub_beam_position[0]
            sub_beam_load_len = sub_beam_position[1] - load_position[0]
            sub_beam_start_load_vector = start_load_vector
            sub_beam_end_load_vector = start_load_vector + sub_beam_load_len*load_diff_vector/load_length
        elif sub_beam_position[1] > load_position[1]:
            sub_beam_start_load_position = load_position[0] - sub_beam_position[0]
            sub_beam_load_len = load_position[1] - load_position[0]
            sub_beam_start_load_vector = start_load_vector
            sub_beam_end_load_vector = end_load_vector
        else:
            return None, None, None, None
    elif load_position[0] <= sub_beam_position[0] < load_position[1]:
        if sub_beam_position[1] < load_position[1]:
            # sub-beam is totally inside load region
            sub_beam_start_load_position = 0.
            sub_beam_load_len = sub_beam_position[1] - sub_beam_position[0]
            sub_beam_start_load_vector = start_load_vector + \
                                         (sub_beam_position[0]-load_position[0])*load_diff_vector/load_length
            sub_beam_end_load_vector = start_load_vector + \
                                       (sub_beam_position[1]-load_position[0])*load_diff_vector/load_length
        else:
            sub_beam_start_load_position = 0.
            sub_beam_load_len = load_position[1] - sub_beam_position[0]
            sub_beam_start_load_vector = start_load_vector + (sub_beam_position[0]-load_position[
                0])*load_diff_vector/load_length
            sub_beam_end_load_vector = end_load_vector
    elif sub_beam_position[0] >= load_position[1]:
        return None, None, None, None
    else:
        assert False
    return sub_beam_start_load_position, sub_beam_load_len, sub_beam_start_load_vector, sub_beam_end_load_vector


def attribute_override(active_attribute, passive_attribute):
    if active_attribute is None:
        flood_att = passive_attribute
    else:
        flood_att = active_attribute

    return flood_att


def get_coordinate_system_idx(load_coord_system):
    system_idx = 0
    if load_coord_system == "GLOB":
        system_idx = 0
    elif load_coord_system == "MEMB":
        system_idx = 1
    return system_idx


def correct_beam_coords_with_offset(beam_coords, joint_labels, beam_offset_infos,
                                    spring_6_dofs_infos, z_ref_joint_coord):
    if joint_labels in beam_offset_infos:
        beam_offset_info = beam_offset_infos[joint_labels]
        if beam_offset_info[0] == "global":
            # convert cm to m
            original_beam_coords = beam_coords.copy()
            for node_idx in range(2):
                offset_vector = np.array(beam_offset_info[node_idx + 1])
                if np.linalg.norm(offset_vector) < 1e-6:
                    continue
                beam_coords[node_idx] += offset_vector
                spring_6_dofs_info = [original_beam_coords[node_idx], beam_coords[node_idx]]
                spring_6_dofs_infos.append(spring_6_dofs_info)
            return beam_coords
        elif beam_offset_info[0] == "local":
            original_beam_coords = beam_coords.copy()
            _rotation_matrix, _translation_vector, _beam_length = \
                find_beam_transformation(beam_coords, z_ref_joint_coord)
            matrix_4x4 = np.identity(4)
            matrix_4x4[:3, :3] = _rotation_matrix
            matrix_4x4[:3, 3] = _translation_vector
            transform = pt.Transformation.from_matrix(matrix_4x4)
            for node_idx in range(2):
                offset_vector = np.array(beam_offset_info[node_idx + 1])
                if np.linalg.norm(offset_vector) < 1e-6:
                    continue
                transformed_offset_vector = transform * pt.Vector(offset_vector)
                beam_coords[node_idx] += np.array(transformed_offset_vector)
                spring_6_dofs_info = [original_beam_coords[node_idx], beam_coords[node_idx]]
                spring_6_dofs_infos.append(spring_6_dofs_info)
            return beam_coords
        else:
            assert False

    return beam_coords


def find_load_case_by_sacs_load_id(sacs_load_id, load_cases):
    for load_case in load_cases:
        load_case_gui_name = load_case.gui_name
        if "-" not in load_case_gui_name:
            continue
        load_case_sacs_id = load_case_gui_name.split("-")[0]
        if load_case_sacs_id == sacs_load_id:

            return load_case
    return None


def _get_center_location(beam):
    point_1 = cu.PortId(beam, 0, 0).get_model_center_point()
    point_2 = cu.PortId(beam, 1, 0).get_model_center_point()
    point = (np.array(point_1) + np.array(point_2)) / 2.0
    return point


def _is_in_range(beam, mgr_ov_infos, seabed_elevation, water_elevation):
    beam_center_loc = _get_center_location(beam)
    z_coordinate = beam_center_loc[2]
    # Check in water elevation and seabed elevation
    check_1 = (seabed_elevation <= z_coordinate) and (z_coordinate <= water_elevation)

    # Check in marine growth range
    check_2 = False
    for marine in mgr_ov_infos:
        if (marine[0] <= z_coordinate) and (z_coordinate <= marine[1]):
            check_2 = True
    # Check whether a beam has tubular cross-section or not
    try:
        API.get_diameter_from_port(beam, 0)
        check_3 = True
    except:
        check_3 = False
    check_circular = check_3

    if check_1:
        if not check_3:
            print ("WARNING: Component ID {} is ignored".format(beam.get_id()))
    check_buoy = check_1 and check_3

    if check_2:
        if not check_3:
            print ("WARNING: Component ID {} is ignored".format(beam.get_id()))
    check_mgr = check_2 and check_3

    return check_buoy, check_mgr, check_circular


def _update_marine_growth(beam, mgr_ov_infos, mgr_ov, check_mgr):
    beam_center_loc = _get_center_location(beam)
    z_coordinate = beam_center_loc[2]
    marine_thickness = 0.0
    marine_density = 0.0

    if mgr_ov is None:
        mgr_att = check_mgr
    elif mgr_ov == 0:
        mgr_att = 0
    else:  # mgr_ov =1 is unsupported
        assert False

    if mgr_att == 1:
        for marine in mgr_ov_infos:
            if (marine[0] <= z_coordinate) and (z_coordinate <= marine[1]):
                marine_thickness = marine[2]*(1e-2)
                marine_density = marine[3]*1000
                break

    return marine_thickness, marine_density


def _assign_linear_wave_load(beam, is_marine_growth, is_flooded, water_density, seabed_elevation, water_elevation,
                             marine_thickness, marine_density, cd_normal, cm_normal, gravity):  # Add
    # params into sources
    # beam.set_parameter("rho_water", water_density)
    # beam.set_parameter("seabed_z", seabed_elevation)
    # beam.set_parameter("water_depth", water_elevation - seabed_elevation)
    beam.set_solver_parameter(1, "hydrodynamics", "morison", "C_d", cd_normal, "real")
    beam.set_solver_parameter(1, "hydrodynamics", "morison", "C_m", cm_normal, "real")

    if is_marine_growth:
        beam.set_solver_parameter(1, "hydrodynamics", "morison", "marine_growth_thickness",
                                               marine_thickness, "real")
        beam.set_solver_parameter(1, "hydrodynamics", "morison", "marine_growth_density",
                                               marine_density * gravity, "real")
    else:
        beam.set_solver_parameter(1, "hydrodynamics", "morison", "marine_growth_thickness",
                                               0, "real")
        beam.set_solver_parameter(1, "hydrodynamics", "morison", "marine_growth_density",
                                               0 * gravity, "real")
    if is_flooded:
        beam.set_solver_parameter(1, "hydrodynamics", "morison", "flooded", is_flooded, "boolean")
    else:
        beam.set_solver_parameter(1, "hydrodynamics", "morison", "flooded", False, "boolean")


def _compute_circle_area(radius):
    return np.pi * radius * radius


def get_buoyancy_force(is_include_marine_growth, is_flooded, radius, thickness, length,
                       marine_thickness, water_density, gravity):
    if is_include_marine_growth == 1:
        outer_radius = radius + marine_thickness
    else:
        outer_radius = radius
    if is_flooded == 1:
        inner_radius = radius - thickness
    else:
        inner_radius = 0.0
    displaced_volume = length * (_compute_circle_area(outer_radius) - _compute_circle_area(inner_radius))
    total_buoyancy_force = water_density * gravity * displaced_volume
    distributed_force = total_buoyancy_force / length
    return distributed_force


def random_words(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def _assign_LIC_plugin(aks_filepath, beams, beam_add_infos, water_density, seabed_elevation,
                       water_elevation, gravity):
    with open(aks_filepath) as file:
        data = json.load(file)

    component_to_apply = beams
    additional_prop = beam_add_infos
    data["component_system"]["plugin_data"]["LIC"] = {}
    data["component_system"]["plugin_data"]["LIC"]["components_parameters"] = {}
    data_comp = data["component_system"]["plugin_data"]["LIC"]["components_parameters"]

    def convert_none_to_zero(content):
        if content is None:
            content = 0.0
        return content

    for component, prop in zip(component_to_apply, additional_prop):
        uuid = component.uuid
        _, _, flood_att, marine_thickness, marine_density, _, cd_normal, cm_normal, _, _ = prop

        data_comp[uuid] = {}
        data_comp[uuid]["boolean_parameters"] = {}
        data_comp[uuid]["file_parameters"] = {}
        data_comp[uuid]["real_parameters"] = {}
        data_comp[uuid]["boolean_parameters"]["flooded"] = flood_att  # in def Create_col (in beam_add_infos)

        data_comp[uuid]["file_parameters"]["additional_data"] = ""
        data_comp[uuid]["real_parameters"]["added_mass_coefficient_normal"] = 1
        data_comp[uuid]["real_parameters"]["drag_coefficient_normal"] = cd_normal
        data_comp[uuid]["real_parameters"]["drag_coefficient_tangential"] = 1
        data_comp[uuid]["real_parameters"]["inertia_coefficient_normal"] = cm_normal
        data_comp[uuid]["real_parameters"]["mg_dens"] = convert_none_to_zero(marine_density)  # in def Create_col (in beam_add_infos)
        data_comp[uuid]["real_parameters"]["mg_thick"] = convert_none_to_zero(marine_thickness)  # in def Create_col (in beam_add_infos)
        data["component_system"]["plugin_data"]["LIC"]["components_parameters"] = data_comp

    # NO LOOP BELOW THIS LINE
    data["component_system"]["plugin_data"]["LIC"]["global_parameters"] = {}
    data_glob = data["component_system"]["plugin_data"]["LIC"]["global_parameters"]
    data_glob["boolean_parameters"] = {}
    data_glob["boolean_parameters"]["buoyancy"] = False
    data_glob["boolean_parameters"]["current_stretch"] = False
    data_glob["boolean_parameters"]["display_sea_surface"] = False
    data_glob["boolean_parameters"]["display_seabed"] = False
    data_glob["boolean_parameters"]["mg_added_mass"] = False
    data_glob["boolean_parameters"]["mg_buoyancy"] = False
    data_glob["boolean_parameters"]["mg_drag"] = False
    data_glob["boolean_parameters"]["mg_inertia"] = False
    data_glob["boolean_parameters"]["particle_acc"] = True
    data_glob["boolean_parameters"]["particle_vel"] = True
    data_glob["boolean_parameters"]["structural_acc"] = False
    data_glob["boolean_parameters"]["structural_vel"] = False
    data_glob["boolean_parameters"]["wave_data_to_text"] = False

    data_glob["file_parameters"] = {}
    data_glob["file_parameters"][
        "additional_data"] = "C:\\Users\\Welcome\\plugins_akselos\\additional_data.json"  # this path is different on each PC

    data_glob["integer_parameters"] = {}
    data_glob["integer_parameters"]["no_current_levels"] = 7  # random value
    data_glob["integer_parameters"]["no_of_comp_jonswap"] = 200.0  # random value
    data_glob["integer_parameters"]["stream_func_order"] = 5.0  # random value
    data_glob["integer_parameters"]["wave_seed_value"] = 12345.0  # random value

    data_glob["real_parameters"] = {}
    data_glob["real_parameters"]["buoy_scale"] = 1.0
    data_glob["real_parameters"]["curr_direction_pl"] = 0.0
    data_glob["real_parameters"]["curr_exponent"] = 7.0
    data_glob["real_parameters"]["curr_speed_seabed"] = 0.0
    data_glob["real_parameters"]["curr_speed_surface"] = 0.0
    data_glob["real_parameters"]["current_dir"] = 0.0
    data_glob["real_parameters"]["current_int_dir"] = 0.0
    data_glob["real_parameters"]["current_int_vel"] = 0.0
    data_glob["real_parameters"]["gamma_value"] = 1.0
    data_glob["real_parameters"]["gravity"] = gravity
    data_glob["real_parameters"]["hydro_dyn_force_scale"] = 1.0
    data_glob["real_parameters"]["iso_joint_scale"] = 1.0
    data_glob["real_parameters"]["max_comp_frq_rng"] = 0.05
    data_glob["real_parameters"]["rel_frq_rng_max"] = 10.0
    data_glob["real_parameters"]["rel_frq_rng_min"] = 0.5
    data_glob["real_parameters"]["sea_dens"] = water_density  # in def Read_model_physic
    data_glob["real_parameters"]["seabed_elevation"] = seabed_elevation  # in def Read_model_physic
    data_glob["real_parameters"]["water_depth"] = water_elevation - seabed_elevation  # in def Read_model_physic
    data_glob["real_parameters"]["wave_dir"] = 0.0
    data_glob["real_parameters"]["wave_height"] = 10.6
    data_glob["real_parameters"]["wave_kin_fac"] = 0.89
    data_glob["real_parameters"]["wave_period"] = 9.7
    data_glob["real_parameters"]["wave_phase"] = 0.0

    data_glob["string_parameters"] = {}
    data_glob["string_parameters"]["current_type"] = "current_interpolated"
    data_glob["string_parameters"]["general_data"] = "general_data"
    data_glob["string_parameters"]["kin_stretch_method"] = "Vertical Stretching"
    data_glob["string_parameters"]["wave_type"] = "airy"
    data["component_system"]["plugin_data"]["LIC"]["global_parameters"] = data_glob

    with open(aks_filepath, 'w') as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
    print ("> Completed assigning parameters into LIC plugin")


def _assign_release_att(aks_filepath, beams, beam_add_infos):
    print ("> Assigning release parameters into AKS model file...")
    with open(aks_filepath) as file:
        data = json.load(file)

    def assign_release_single_port(index, data, release, component_id):
        for port_constraint_idx, port_constraint in enumerate(data["component_system"]["port_constraints"]):
            if port_constraint_idx == index:
                bool_string = "local_" + str(component_id)
                release_string = "release_" + str(component_id)
                port_constraint["boolean_data"][bool_string] = False
                port_constraint["string_data"][release_string] = release
                break

    def create_release_layout(index, list, data, release, component_id):
        for port_constraint_idx, port_constraint in enumerate(data["component_system"]["port_constraints"]):
            if port_constraint_idx == index:
                port_constraint["string_data"] = {}
                for l in list:
                    bool_string = "local_" + str(l)
                    release_string = "release_" + str(l)
                    port_constraint["boolean_data"][bool_string] = False
                    port_constraint["string_data"][release_string] = "000000"
                break
        assign_release_single_port(index, data, release, component_id)

    def assign_release_att(data, component_id, port_index, release):
        for port_constraint_idx, port_constraint in \
                enumerate(data["component_system"]["port_constraints"]):
            component_id_list = []
            found_idx = None
            # Because "port_constraints" is a LIST not a DICTIONARY, cannot access directly using keywords,
            # must use loop to access data inside "port_constraints"
            for port_id_data in port_constraint["port_id_data"]:
                component_id_list.append(port_id_data["component_id"])
                if not port_id_data["component_id"] == component_id or \
                        not port_id_data["port_index"] == port_index:
                    continue
                found_idx = port_constraint_idx

            if found_idx is not None:
                if port_constraint["boolean_data"] == {}:
                    create_release_layout(found_idx, component_id_list, data, release, component_id)
                else:
                    assign_release_single_port(found_idx, data, release, component_id)
                break
    # Main
    component_to_apply = beams
    additional_prop = beam_add_infos
    for component, prop in zip(component_to_apply, additional_prop):
        _, _, _, _, _, _, _, _, release_0, release_1 = prop
        component_id = component.get_id()

        if release_0 == "" or release_1 == "":
            continue

        if not float(release_0) == 0:
            node_id = 0
            assign_release_att(data, component_id, node_id, release_0)

        if not float(release_1) == 0:
            node_id = 1
            assign_release_att(data, component_id, node_id, release_1)

    with open(aks_filepath, 'w') as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
    print ("> Completed assigning release parameters into AKS model file !")


def modify_master_port(aks_filepath, beams, beam_add_infos):
    print ("> Modifying master port...")
    with open(aks_filepath) as file:
        data = json.load(file)

    def flip_data(data, component_id, node_id, comp_id_list, index):
        _tmp_port_idx = 0
        location_replace = 0
        component_id_master = 0
        port_index_master = 0
        for port_constraint_idx, port_constraint in \
                enumerate(data["component_system"]["port_constraints"]):
            if not port_constraint_idx == index:
                continue

            for comp_id in comp_id_list:
                release_string = "release_" + str(comp_id)
                if port_constraint["string_data"][release_string] == "000000":
                    _tmp_port_idx = comp_id
                    break
            # LOOP 1: Save data of port with release == "000000"
            for port_id_idx, port_id in enumerate(port_constraint["port_id_data"]):
                if port_id["component_id"] == _tmp_port_idx:
                    location_replace = port_id_idx
                    component_id_master = _tmp_port_idx
                    port_index_master = port_id["port_index"]
            # LOOP 2: Flip info
            for port_id_idx, port_id in enumerate(port_constraint["port_id_data"]):
                if port_id_idx == 0:
                    port_id["component_id"] = component_id_master
                    port_id["port_index"] = port_index_master
                if port_id_idx == location_replace:
                    port_id["component_id"] = component_id
                    port_id["port_index"] = node_id

    def check_master_loc(data, component_id, node_id):
        for port_constraint_idx, port_constraint in \
                enumerate(data["component_system"]["port_constraints"]):
            component_id_list = []
            found_idx = None
            for port_id_idx, port_id in enumerate(port_constraint["port_id_data"]):
                component_id_list.append(port_id["component_id"])
                if port_id["component_id"] == component_id and \
                        port_id["port_index"] == node_id and port_id_idx == 0:
                    found_idx = port_constraint_idx
            if found_idx is not None:
                flip_data(data, component_id, node_id, component_id_list, found_idx)
                break

    # Main
    component_to_apply = beams
    additional_prop = beam_add_infos
    for component, prop in zip(component_to_apply, additional_prop):
        _, _, _, _, _, _, _, _, release_0, release_1 = prop
        component_id = component.get_id()
        if release_0 == "" or release_1 == "":
            continue

        if not float(release_0) == 0:
            node_id = 0
            check_master_loc(data, component_id, node_id)

        if not float(release_1) == 0:
            node_id = 1
            check_master_loc(data, component_id, node_id)

    with open(aks_filepath, 'w') as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
    print ("> Completed Modifying master port !")


def calculate_drag_mass_coef(beam, check_mgr, cdm_infos, check_circular):
    d = []
    cd_normal_clean = []
    cm_normal_clean = []
    cd_normal_foul = []
    cm_normal_foul = []
    cd_normal = 1  # initial value
    cm_normal = 1  # initial value

    for cdm in cdm_infos:
        dia, cd_n_c, cm_n_c, cd_n_f, cm_n_f = cdm
        d.append(dia)
        cd_normal_clean.append(cd_n_c)
        cm_normal_clean.append(cm_n_c)
        cd_normal_foul.append(cd_n_f)
        cm_normal_foul.append(cm_n_f)

    if check_circular:
        diameter, _ = API.get_diameter_from_port(beam, 0)
        for x in range(0, len(cdm_infos)-1):
            if (d[x] <= diameter) and (diameter <= d[x+1]):
                if check_mgr:  # Fouled member (check_mgr = True -> marine growth available)
                    cd_normal = np.interp(diameter, [d[x], d[x+1]], [cd_normal_foul[x], cd_normal_foul[x+1]])
                    cm_normal = np.interp(diameter, [d[x], d[x+1]], [cm_normal_foul[x], cm_normal_foul[x+1]])
                else:  # Clean member
                    cd_normal = np.interp(diameter, [d[x], d[x+1]], [cd_normal_clean[x], cd_normal_clean[x+1]])
                    cm_normal = np.interp(diameter, [d[x], d[x+1]], [cm_normal_clean[x], cm_normal_clean[x+1]])

    return cd_normal, cm_normal


def cone_diameter_interp(parent_beam_length, sub_beam_length, dimensions_ori, i):

    mid_point = (i * sub_beam_length + 0.5 * sub_beam_length)
    root_to_small_dia = (dimensions_ori[2] * parent_beam_length) / (dimensions_ori[0] - dimensions_ori[2])
    diameter = (dimensions_ori[0] - dimensions_ori[2]) * (mid_point + root_to_small_dia) / parent_beam_length

    # Re-assign dimensions array
    dimensions = [0, 0, 0, 0]
    dimensions[0] = diameter
    dimensions[1] = dimensions_ori[1]
    return dimensions


def create_collection_and_model(collection_name, node_coords, shell_mesh_files, beam_elem_infos,
                                model_info, cross_section_mesh_path, external_sect_lib):
    print("> Create Collection: ")
    working_collection_path = os.path.join(directories.DATA_DIR, 'collections', collection_name)
    print("Working Collection path {}".format(working_collection_path))
    #component_stiffener_info = model_info['component_stiffener_info']
    sect_infos = model_info['sect_infos']
    grup_infos = model_info['grup_infos']
    node_labels = model_info['node_labels']
    load_infos = model_info['load_infos']
    load_combination_infos = model_info['load_combination_infos']
    constraint_nodes = model_info['constraint_nodes']
    linear_spring_nodes = model_info['linear_spring_nodes']
    all_beams_have_load = model_info['all_beams_have_load']
    beam_offset_infos = model_info['beam_offset_infos']
    group_ov_infos = model_info['grp_ov_infos']
    mem_ov_infos = model_info['mem_ov_infos']
    mgr_ov_infos = model_info['mgr_ov_infos']
    water_density = model_info['water_density']
    seabed_elevation = model_info['seabed_elevation']
    water_elevation = model_info['water_elevation']
    cdm_infos = model_info['cdm_infos']

    delete_dirs = ['components', 'ports', 'aks_files', 'cross_sections', 'nlspring_data']
    for dir in delete_dirs:
        del_dir = os.path.join(working_collection_path, dir)
        if os.path.exists(del_dir):
            print ("> Removing", del_dir)
            shutil.rmtree(del_dir)

    collection = td.Collection(collection_name)
    dimensional_scaling = collection.get_dimensional_variables()
    accel_0 = dimensional_scaling["accel_0"].value
    ap.update_predefined_beam_material_parameter_group(collection)
    operator_name_to_type = collection.get_physics_type().get_operator_name_to_type()
    component_system = cs.RBComponentSystem(collection)
    component_id = 0

    # SHELL
    default_shell_thickness = 0.05
    use_material_in_mesh = False
    subdirectory = 'components'
    extra_nodes = ('',)
    #surface_load_ids_range = {'min': 200, 'max': 299}
    node_port_ids_range = {'min': 500, 'max': 2999}
    #physics_type = collection.get_physics_type()
    shells = []

    #block_attributes = component_block_attributes[comp_idx]
    #all_beam_joint_labels = set([l for l, _, _, _, _ in beam_elem_infos])
    free_beam_infos = beam_elem_infos
    print("> Create shell components")
    all_collection_material_value_groups = {}
    stiffener_joint_labels = {}
    stiffener_for_del = []
    
    if CREATE_SHELL_COMPONENTS:
        for comp_idx, shell_mesh_file in enumerate(shell_mesh_files):
            # if not comp_idx == 0:
            #     continue
            print(" > Create component from", shell_mesh_file)
            # progress_bar = ap.ProgressBar(10+len(free_beam_infos))
            progress_bar = ap.ProgressBar(10)
            progress_bar.update(1)

            ref_component_type, block_attributes = new_type.create_new_ref_component_type(
                shell_mesh_file, collection, default_shell_thickness,
                use_material_in_mesh, extra_nodes=extra_nodes, subdirectory=subdirectory,
                create_material_group_box=True)  # Return corresponding Component type

            class id_param:  # Dummy class
                def __init__(self):
                    self.port_ids_range = {"min": 100, "max": 199}
                    self.normalload_ids_range = {"min": 200, "max": 249}
                    self.surfaceload_ids_range = {"min": 250, "max": 299}
                    self.dirichlet_ids_range = {"min": 300, "max": 399}
                    self.hydrostatic_ids_range = {"min": 400, "max": 429}
                    self.spring_x_ids_range = {"min": 430, "max": 459}
                    self.spring_y_ids_range = {"min": 460, "max": 479}
                    self.spring_z_ids_range = {"min": 480, "max": 499}
                    self.contact_range = {"min": 500, "max": 549}

            ru.RefComponentTypeGeomUI.auto_assign_sideset_using_id_convention(ref_component_type,
                use_non_conforming=False, relative_tol=1e-3, allow_scaling=False, summary_box=None,
                id_parameters=id_param(), updater=None, extract_missing_port=True, add_free_nodeset_to_mesh=False,
                add_extra_point_to_edge_port=True)


            gui_parameter_group_boxes = ref_component_type.core.gui_parameter_group_boxes
            gui_additional_value_groups = ref_component_type.core.gui_additional_value_groups
            for ga in gui_additional_value_groups:
                value_key = tuple(ga.values)
                if value_key not in all_collection_material_value_groups:
                    all_collection_material_value_groups[value_key] = ga.name
                else:
                    # all_collection_material_value_groups[value_key] = ga.name
                    replaced_name = ga.name
                    name_to_replace = all_collection_material_value_groups[value_key]
                    for gp in ref_component_type.core.gui_parameter_group_boxes:
                        if gp.default_value_group == replaced_name:
                            gp.default_value_group = name_to_replace
                            gp.gui_name = name_to_replace

            progress_bar.update(7)
            #os.unlink(shell_mesh_file)

            # Assign node port to nodeset
            mesh = ref_component_type.get_default_mesh()
            block_id_to_names = mesh.mesh_data.block_id_to_names
            for nodeset_id, node_idxs in mesh.labeled_vertices.items():
                if node_port_ids_range["min"] <= nodeset_id.mesh_id <= node_port_ids_range["max"]:
                    ref_component_type.add_single_node_port(nodeset_id, node_idxs)
            # Assign stiffener to 1D block
            exo_data = mesh.mesh_data  # ExoData instance
            all_shell_elems = []
            for block_idx, exo_block_data in enumerate(exo_data.exo_block_datas):
                if not exo_block_data.elem_type.is_shell:
                    continue

                all_shell_elems.append(exo_block_data.elems)
            all_shell_elems = np.concatenate(all_shell_elems)

            # block_id_to_names = exo_data.block_id_to_names
            #assert block_id_to_names is not None
            stiffener_mass_density_source_map = {}
            subdomain_ids_for_stiffener_selfweight = []
            collection_type = ref_component_type.get_collection_type()
            store_mass_density = 0
            density_diff = 0
            for exo_block in exo_data.exo_block_datas:
                if exo_block.elem_type.dimension != 1:
                    continue
                beam_block_id = exo_data.block_idx_to_subdomain_id[exo_block.block_idx]  # need to add suffix for block id to distinguish sub-beams
                beam_block_name = ref_component_type.get_subdomain_name(beam_block_id)

                if beam_block_name is None:
                    beam_block_name = str(beam_block_id)
                items = beam_block_name.split('_')
                _joint_labels = items[0].split('-')[1]
                grup_name = items[1]
                _joint_0_label = _joint_labels[:len(_joint_labels) // 2].strip()
                _joint_1_label = _joint_labels[len(_joint_labels) // 2:].strip()
                _node_0_idx = node_labels[_joint_0_label]
                _node_1_idx = node_labels[_joint_1_label]
                _beam_elem = [_node_0_idx, _node_1_idx]
                beam_coords = node_coords[_beam_elem, :]
                _, _, _beam_length = find_beam_transformation(beam_coords)

                stiffener_for_del.append(_joint_labels)
                # _beam_length = float('.'.join(items[2:4]))
                # TODO: get more data from grup_infos

                sect_label, grup_dimensions, young_modulus, mass_density, _, _, poisson_ratio = grup_infos[
                    grup_name]
                if sect_label in sect_infos:
                    sect_type, dimensions = sect_infos[sect_label]
                else:
                    sect_type = sect_label
                    dimensions = grup_dimensions
                if sect_type == 'CON':
                    # TODO THUC: handle CON
                    assert False
                mesh_beam_name = sect_label + '_' + sect_type + ".exo"
                if sect_type == "":
                    mesh_beam_name = get_regular_tube(dimensions)
                cross_section_mesh = os.path.join(cross_section_mesh_path, mesh_beam_name)
                if not os.path.exists(cross_section_mesh):
                    # It's not SACS builtin cross-section, find it in external library
                    try:
                        mesh_beam_name, _ = get_external_defined_xsecs(
                            sect_type, external_sect_lib, return_stiffener_mesh=True)
                        cross_section_mesh = os.path.join(cross_section_mesh_path, mesh_beam_name)
                    except Exception as e:
                        print(e)
                        assert False

                    if not os.path.exists(cross_section_mesh):
                        assert False

                first_stiffener_elem = exo_block.elems[0]
                beam_coords = exo_data.coords[first_stiffener_elem]
                beam_axis, beam_t1, beam_t2 = calculate_stiffener_orientations(
                    first_stiffener_elem, exo_data.coords)
                young_modulus_GPa = young_modulus

                t1_t2_offset = [0., 0.]
                if _joint_labels in beam_offset_infos:
                    beam_offset_info = beam_offset_infos[_joint_labels]
                    offset_type, joint_0_offset, joint_1_offset = beam_offset_info
                    if offset_type == "local":
                        joint_0_offset = np.array(joint_0_offset)
                        joint_1_offset = np.array(joint_1_offset)
                        offset_vector = 0.5 * (joint_0_offset + joint_1_offset)
                        t1_t2_offset[0] = offset_vector[1]
                        t1_t2_offset[1] = offset_vector[2]

                    elif offset_type == "global":
                        _rotation_matrix, _translation_vector, _ = \
                            find_beam_transformation(beam_coords)
                        matrix_4x4 = np.identity(4)
                        matrix_4x4[:3, :3] = _rotation_matrix
                        matrix_4x4[:3, 3] = _translation_vector
                        transform = pt.Transformation.from_matrix(matrix_4x4).inv()
                        offset_vector = np.zeros(3)
                        for node_idx in range(2):
                            local_offset_vector = \
                                transform * pt.Vector(np.array(beam_offset_info[node_idx + 1]))
                            offset_vector += np.array(local_offset_vector)
                        offset_vector *= 0.5
                        t1_t2_offset[0] = offset_vector[1]
                        t1_t2_offset[-1] = offset_vector[2]
                    if np.abs(t1_t2_offset[0]) < 1e-4:
                        t1_t2_offset[0] = 0.
                    if np.abs(t1_t2_offset[1]) < 1e-4:
                        t1_t2_offset[1] = 0.

                nondimensional_E = dv.decode_param(
                    collection, "", young_modulus_GPa, type="young_modulus")
                scalars_values = {
                    'young_modulus': str(nondimensional_E),
                    'poisson_ratio': str(poisson_ratio),
                    'stiffener_offset_orthogonal_to_shell_normal': str(t1_t2_offset[-1]),
                # Trong: swap offset value
                    'stiffener_offset_along_shell_normal': str(t1_t2_offset[0]),
                    'stiffener_flip_orientation': '0.0',
                    'stiffener_angle_degrees': '0.0'
                }
                beam_t2_axis = beam_t2
                cross_section_name = 'cross_sections/' + mesh_beam_name.replace('.exo', '')

                if cross_section_name in collection.rb_ref_components:
                    pass
                else:
                    # print (" + Creating new beam type", cross_section_name)
                    t_beam_type = new_type.new_cross_section_type(cross_section_mesh, collection)
                    ap.write_type_to_json(t_beam_type, directories.DATA_DIR)

                operator_type_name = "elasticity_beam_stiffener"
                ref_component_type.add_stiffener_type_to_1d_block(
                    operator_type_name, beam_block_id,
                    cross_section_name + '/' + mesh_beam_name.replace('.exo', ''),
                    scalars_values, beam_t2_axis=beam_t2_axis, subdomain_name=beam_block_name)
                stiffener_joint_labels[_joint_labels] = (component_id, beam_block_id, _beam_length)

                if not ENABLE_AVERAGE_DENSITY_FOR_STIFF:
                    # add "Self-weight" source for stiffener to each component_type
                    sw_operator_type = operator_name_to_type.get(ot.SELF_WEIGHT, None)
                    sw_source_type = ref_component_type.add_new_source(
                        sw_operator_type, source_gui_name="Self_weight_" + str(beam_block_id))
                    operator_id = oi.SourceOperatorId(ref_component_type, sw_source_type.core.name)
                    operator_id.set_mesh_ids([beam_block_id])
                    source_params = sw_source_type.get_parameters()
                    assert len(source_params) == 1, len(source_params)

                    mass_density_param_name = source_params[0].name
                    # set mass density value for new created Self-weight
                    # mass_density = MASS_DENSITY_SCALING * mass_density

                    nondimensional_mass_density = dv.decode_param(
                        collection, "", mass_density, type="mass_density")
                    stiffener_mass_density_source_map[mass_density_param_name] = \
                        nondimensional_mass_density

                    # Add 'gui_parameter_group_box' for stiffener block
                    value_key = tuple([str(mass_density) + " kg/m^3",
                                       str(young_modulus_GPa) + " GPa",
                                       str(0.3)])
                    if value_key in all_collection_material_value_groups:
                        value_group_name = all_collection_material_value_groups[value_key]
                    else:
                        value_group_name = grup_name
                        all_collection_material_value_groups[value_key] = value_group_name

                    found = False
                    for gp in gui_parameter_group_boxes:
                        # compare material values
                        if gp.default_value_group == value_group_name:
                            found = True
                            gp.subdomains.append(beam_block_name)
                    if not found:
                        gui_parameter_group_box = tp.GuiParameterGroupBox(
                            default_value_group=value_group_name,
                            gui_name=value_group_name,
                            parameter_group_name="material_properties",
                            subdomains=[beam_block_name],
                            value_group_list=collection.name + "_materials")
                        gui_parameter_group_boxes.append(gui_parameter_group_box)

                if store_mass_density == 0:
                    store_mass_density += mass_density
                else:
                    store_mass_density += mass_density
                    store_mass_density = store_mass_density/2
                    density_diff_current = np.abs(store_mass_density - mass_density)
                    if density_diff_current > density_diff:
                        density_diff = density_diff_current


                # Add "beam" load source to stiffener
                if CREATE_ALL_SOURCE_FOR_STIFF:  # Create all sources for all stiffeners
                    # add CONC source
                    conc_operator_type = operator_name_to_type.get(
                        ot.STIFFENER_CONCENTRATED_LOAD, None)
                    conc_source_type = ref_component_type.add_new_source(
                        conc_operator_type, source_gui_name="stiffener_concentrated_" + str(
                            beam_block_id))
                    operator_id = oi.SourceOperatorId(ref_component_type, conc_source_type.core.name)
                    operator_id.set_mesh_ids([beam_block_id])
                    # add UNIF source
                    ld_operator_type = operator_name_to_type.get(
                        ot.STIFFENER_LINEAR_DISTRIBUTED_LOAD, None)
                    ld_source_type = ref_component_type.add_new_source(
                        ld_operator_type, source_gui_name="stiffener_linear_distributed_" + str(
                            beam_block_id))
                    operator_id = oi.SourceOperatorId(ref_component_type, ld_source_type.core.name)
                    operator_id.set_mesh_ids([beam_block_id])
                    # add PART-UNIF source
                    pd_operator_type = operator_name_to_type.get(
                        ot.STIFFENER_PARTIAL_DISTRIBUTED_LOAD, None)
                    pd_source_type = ref_component_type.add_new_source(
                        pd_operator_type, source_gui_name="`" + str(
                            beam_block_id))
                    operator_id = oi.SourceOperatorId(ref_component_type, pd_source_type.core.name)
                    operator_id.set_mesh_ids([beam_block_id])
                else:
                    if "CONC" in all_beams_have_load:
                        if _joint_labels in all_beams_have_load["CONC"]:
                            conc_operator_type = operator_name_to_type.get(
                                ot.STIFFENER_CONCENTRATED_LOAD, None)
                            conc_source_type = ref_component_type.add_new_source(
                                conc_operator_type, source_gui_name="stiffener_concentrated_" + str(
                                    beam_block_id))
                            operator_id = oi.SourceOperatorId(ref_component_type, conc_source_type.core.name)
                            operator_id.set_mesh_ids([beam_block_id])
                    if "UNIF" in all_beams_have_load:
                        if _joint_labels in all_beams_have_load["UNIF"]:
                            ld_operator_type = operator_name_to_type.get(
                                ot.STIFFENER_LINEAR_DISTRIBUTED_LOAD, None)
                            ld_source_type = ref_component_type.add_new_source(
                                ld_operator_type, source_gui_name="stiffener_linear_distributed_" + str(
                                    beam_block_id))
                            operator_id = oi.SourceOperatorId(ref_component_type, ld_source_type.core.name)
                            operator_id.set_mesh_ids([beam_block_id])
                    if "PART-UNIF" in all_beams_have_load:
                        if _joint_labels in all_beams_have_load["PART-UNIF"]:
                            pd_operator_type = operator_name_to_type.get(
                                ot.STIFFENER_PARTIAL_DISTRIBUTED_LOAD, None)
                            pd_source_type = ref_component_type.add_new_source(
                                pd_operator_type, source_gui_name="`" + str(
                                    beam_block_id))
                            operator_id = oi.SourceOperatorId(ref_component_type, pd_source_type.core.name)
                            operator_id.set_mesh_ids([beam_block_id])

                ref_component_type.clear_method_cache()


            # add Self-weight source
            if ENABLE_AVERAGE_DENSITY_FOR_STIFF:
                for beam_block_geom_id in mesh.get_1d_subdomain_geom_ids():
                    beam_block_id = beam_block_geom_id.mesh_id
                    mesh_subdomain_name = block_id_to_names.get(beam_block_id, str(beam_block_id))
                    json_subdomain_name = ref_component_type.get_subdomain_name(beam_block_geom_id)
                    block_attribute = block_attributes.get(beam_block_id, None)

                    is_stiff = ru.RefComponentTypeGeomUI.add_stiffener_type_to_1d_block(
                        ref_component_type, beam_block_id, mesh_subdomain_name,
                        json_subdomain_name, block_attribute, collection_type,
                        summary_box=None)
                    if is_stiff:
                        subdomain_ids_for_stiffener_selfweight.append(beam_block_id)
                # print('subdomain_ids_for_stiffener_selfweight', subdomain_ids_for_stiffener_selfweight)

                # Add self-weight source to stiffener
                if len(subdomain_ids_for_stiffener_selfweight) > 0:
                    physics_type = collection_type.get_physics_type()
                    operator_name_to_type = physics_type.get_operator_name_to_type()
                    operator_type = operator_name_to_type[ot.SELF_WEIGHT]
                    sw_source_type = ref_component_type.add_new_source(
                        operator_type, source_gui_name="Self_weight_for_stiffeners")
                    operator_id = oi.SourceOperatorId(ref_component_type, sw_source_type.core.name)
                    operator_id.set_mesh_ids(subdomain_ids_for_stiffener_selfweight)
                    source_params = sw_source_type.get_parameters()
                    assert len(source_params) == 1, len(source_params)

                    mass_density_param_name = source_params[0].name
                    # set mass density value for new created Self-weight
                    # mass_density = MASS_DENSITY_SCALING * mass_density

                    nondimensional_mass_density = dv.decode_param(
                        collection, "", store_mass_density, type="mass_density")
                    stiffener_mass_density_source_map[mass_density_param_name] = \
                        nondimensional_mass_density
                if density_diff > STIFF_DENSITY_TOL:
                    print('CAUTION: Average mass density value for stiffeners in component {} exceeds allowable value'.format(comp_idx))

            ref_component_type.core.gui_parameter_group_boxes = gui_parameter_group_boxes
            ref_component_type.core.gui_additional_value_groups = []
            # progress_bar.update(4280)
            progress_bar.update(2)

            ap.write_type_to_json(ref_component_type, directories.DATA_DIR)
            shell_component = rb.RBComponent(component_id, ref_component_type)
            shell_component.post_init()
            for mass_density_param_name, value in stiffener_mass_density_source_map.items():
                shell_component.mu_coeff[mass_density_param_name].value = value
            shells.append(shell_component)
            component_id += 1

    if len(all_collection_material_value_groups) == 0:
        # Write all used materials to collection.json
        value_key = tuple(["7850 kg/m^3", "200 GPa", "0.3"])
        all_collection_material_value_groups[value_key] = "Generic Steel"
    ap.write_material_value_groups_to_collection_type(
        collection, all_collection_material_value_groups)

    # BEAM
    print("> Create beam components")
    beams = []
    beam_add_infos = []
    linear_springs = []
    if IS_USE_RIGID is True:
        six_dofs_spring = collection.rb_ref_components['/builtin/rigid_beam']
    else:
        assert False
        # six_dofs_spring = collection.rb_ref_components['/builtin/six_dofs_spring']

    beam_label_to_comp_id = {}
    spring_6_dofs_infos = []
    progress_bar = ap.ProgressBar(len(free_beam_infos))
    # new_beam_infos = split_beams(free_beam_infos, split_beam_infos)
    for beam_info in free_beam_infos:
        joint_labels, grup_name, beam_elem, rotation_angle, flood_att_mem, z_ref_joint, release_0, release_1 = beam_info
        if joint_labels in stiffener_for_del:
            continue

        for member in mem_ov_infos:
            _, _, beam_nodes, _, _ = member
            if beam_nodes == beam_elem:
                mem_ov_info = member
                break
        else:
            mem_ov_info = [None, None, None, None, None]  # Trong: fix bug access before assignment

        for grp in group_ov_infos:
            if grp == grup_name:
                grp_ov_info = group_ov_infos[grp]
                break
        else:
            grp_ov_info = [None, None, None, None, None, None, None]

        grup_info = grup_infos[grup_name]
        beam_coords = node_coords[beam_elem, :]

        if z_ref_joint is None:
            z_ref_joint_coord = None
        else:
            z_ref_joint_coord = node_coords[z_ref_joint] # z_ref_joint is node index in node_coords

        if not isinstance(grup_info, list):  # IF this beam does NOT have segments
            sect_label, grup_dimensions, young_modulus, mass_density, sub_beam_length, \
                flood_att_gr, poisson_ratio = grup_info
            if sect_label in sect_infos:
                sect_type, dimensions = sect_infos[sect_label]
            else:
                sect_type = sect_label
                dimensions = grup_dimensions

            if not sect_type == 'CON':  # Single beam without CON (CONE) cross-section
                grup_ov_name, mgr_ov, flood_att_grp_ov, xsec_ov, dis_area_ov, cd_ov_gp, cm_ov_gp = grp_ov_info
                _, flood_att_mem_ov, _, cd_ov_m, cm_ov_m = mem_ov_info
                if xsec_ov == 0.001:
                    mass_density = 0

                flood_att_original = attribute_override(flood_att_mem, flood_att_gr) #attribute_override(1,2) : 1 overrides 2
                flood_att_ov = attribute_override(flood_att_mem_ov, flood_att_grp_ov)
                flood_att = attribute_override(flood_att_ov, flood_att_original)
                cd_ov = attribute_override(cd_ov_m, cd_ov_gp)
                cm_ov = attribute_override(cm_ov_m, cm_ov_gp)

                beam_coords = correct_beam_coords_with_offset(
                    beam_coords, joint_labels, beam_offset_infos, spring_6_dofs_infos, z_ref_joint_coord)
                beam, n_linear_springs = create_beam(
                    component_id, beam_coords, beam_elem, joint_labels, sect_type,
                    dimensions, sect_label, rotation_angle, collection, young_modulus, mass_density,
                    cross_section_mesh_path, external_sect_lib, beam_label_to_comp_id, node_labels,
                    constraint_nodes, linear_spring_nodes, linear_springs, flood_att, poisson_ratio,
                    z_ref_joint_coord)
                component_id += 1 + n_linear_springs
                beams.append(beam)

                check_buoy, check_mgr, check_circular = _is_in_range(beam, mgr_ov_infos, seabed_elevation, water_elevation)
                marine_thickness, marine_density = _update_marine_growth(beam, mgr_ov_infos, mgr_ov, check_mgr)
                cd_normal, cm_normal = calculate_drag_mass_coef(beam, check_mgr, cdm_infos, check_circular)
                cd_normal = attribute_override(cd_ov, cd_normal)
                cm_normal = attribute_override(cm_ov, cm_normal)
                # _assign_linear_wave_load(beam, check_mgr, flood_att, water_density, seabed_elevation, water_elevation,
                #                          marine_thickness, marine_density, cd_normal, cm_normal, accel_0)
                beam_add_info = (
                check_buoy, check_mgr, flood_att, marine_thickness, marine_density, dis_area_ov, cd_normal, cm_normal,
                release_0, release_1)

                beam_add_infos.append(beam_add_info)
            else:  # Single beam with CON (CONE) cross-section
                offset_beam_coords = correct_beam_coords_with_offset(
                    beam_coords, joint_labels, beam_offset_infos, spring_6_dofs_infos, z_ref_joint_coord)

                beam_vector = offset_beam_coords[1] - offset_beam_coords[0]
                parent_beam_length = np.linalg.norm(beam_vector)  # include offset
                beam_vector_unit = beam_vector / parent_beam_length

                new_grup_info = []
                cone_sub_beam = 10
                sub_beam_len = parent_beam_length / cone_sub_beam # Different with

                for i in range(cone_sub_beam, 0, -1):
                    sect_type = 'TUB'
                    sub_dimensions = cone_diameter_interp(parent_beam_length, sub_beam_len, dimensions, i)
                    sub_grup_info = (sect_label, grup_dimensions, young_modulus, mass_density,
                                     sub_beam_len, flood_att_gr, poisson_ratio, sect_type, sub_dimensions)
                    new_grup_info.append(sub_grup_info)

                grup_info = new_grup_info

                sub_beam_lengths = []
                for sub_beam_idx, member_info in enumerate(grup_info):
                    _, _, _, _, sub_beam_length, _, _, _, _ = member_info
                    sub_beam_lengths.append(sub_beam_length)

                sub_beam_coords_array = np.empty((len(grup_info), 2, 3))
                sub_beam_coords_array[:] = np.nan
                sub_beam_coords_array[0, 0, :] = offset_beam_coords[0]
                sub_beam_coords_array[-1, -1, :] = offset_beam_coords[-1]
                for sub_beam_idx, sub_beam_length in enumerate(sub_beam_lengths):
                    if sub_beam_length is None:
                        break
                    if np.isnan(sub_beam_coords_array[sub_beam_idx, 0, 0]):
                        break
                    sub_beam_coords_array[sub_beam_idx, 1, :] = \
                        sub_beam_coords_array[sub_beam_idx, 0, :] + sub_beam_length * beam_vector_unit
                    if sub_beam_idx < len(sub_beam_lengths) - 1:
                        sub_beam_coords_array[sub_beam_idx + 1, 0, :] = sub_beam_coords_array[sub_beam_idx, 1, :]
                for sub_beam_idx, sub_beam_length in reversed(list(enumerate(sub_beam_lengths))):
                    if sub_beam_length is None:
                        break
                    if np.isnan(sub_beam_coords_array[sub_beam_idx, 1, 0]):
                        break
                    sub_beam_coords_array[sub_beam_idx, 0, :] = \
                        sub_beam_coords_array[sub_beam_idx, 1, :] - sub_beam_length * beam_vector_unit
                    if sub_beam_idx > 0:
                        sub_beam_coords_array[sub_beam_idx - 1, 1, :] = sub_beam_coords_array[sub_beam_idx, 0, :]
                assert not np.any(np.isnan(sub_beam_coords_array)), sub_beam_coords_array

                beam_label_to_comp_id[joint_labels] = []
                for sub_beam_idx, member_info in enumerate(grup_info):
                    sect_label, grup_dimensions, young_modulus, mass_density, _, \
                    flood_att_gr, poisson_ratio, sect_type, dimensions = member_info
                    grup_ov_name, mgr_ov, flood_att_grp_ov, xsec_ov, dis_area_ov, cd_ov_gp, cm_ov_gp = grp_ov_info
                    _, flood_att_mem_ov, _, cd_ov_m, cm_ov_m = mem_ov_info
                    if xsec_ov == 0.001:
                        mass_density = 0.0
                    sub_beam_length = sub_beam_lengths[sub_beam_idx]
                    sub_beam_coords = sub_beam_coords_array[sub_beam_idx]
                    beam_label_to_comp_id[joint_labels].append((component_id, sub_beam_length))
                    flood_att_original = attribute_override(flood_att_mem, flood_att_gr)
                    flood_att_ov = attribute_override(flood_att_mem_ov, flood_att_grp_ov)
                    flood_att = attribute_override(flood_att_ov, flood_att_original)
                    cd_ov = attribute_override(cd_ov_m, cd_ov_gp)
                    cm_ov = attribute_override(cm_ov_m, cm_ov_gp)

                    beam, n_linear_springs = create_beam(
                        component_id, sub_beam_coords, beam_elem, joint_labels, sect_type, dimensions,
                        sect_label, rotation_angle, collection, young_modulus, mass_density,
                        cross_section_mesh_path, external_sect_lib, beam_label_to_comp_id, node_labels,
                        constraint_nodes, linear_spring_nodes, linear_springs, flood_att, poisson_ratio,
                        z_ref_joint_coord, sub_beam_idx=sub_beam_idx)
                    component_id += 1 + n_linear_springs
                    beams.append(beam)

                    check_buoy, check_mgr, check_circular = _is_in_range(beam, mgr_ov_infos, seabed_elevation,
                                                                         water_elevation)
                    marine_thickness, marine_density = _update_marine_growth(beam, mgr_ov_infos, mgr_ov, check_mgr)
                    cd_normal, cm_normal = calculate_drag_mass_coef(beam, check_mgr, cdm_infos, check_circular)
                    cd_normal = attribute_override(cd_ov, cd_normal)
                    cm_normal = attribute_override(cm_ov, cm_normal)
                    # THE BELOW FUNCTION IS ONLY TEMPORARILY DISABLED (DO NOT DELETE) !
                    # _assign_linear_wave_load(
                    #     beam, check_mgr, flood_att, water_density, seabed_elevation, water_elevation,  marine_thickness,
                    #     marine_density, cd_normal, cm_normal, accel_0)
                    beam_add_info = (
                        check_buoy, check_mgr, flood_att, marine_thickness, marine_density, dis_area_ov, cd_normal,
                        cm_normal,
                        release_0, release_1)
                    beam_add_infos.append(beam_add_info)

        else:      # Members with sub-beams
            sub_beam_lengths = []
            for sub_beam_idx, member_info in enumerate(grup_info):
                sect_label, _, _, _, sub_beam_length, _, _ = member_info
                sub_beam_lengths.append(sub_beam_length)

            offset_beam_coords = correct_beam_coords_with_offset(
                beam_coords, joint_labels, beam_offset_infos, spring_6_dofs_infos, z_ref_joint_coord)
            beam_vector = offset_beam_coords[1] - offset_beam_coords[0]
            parent_beam_length = np.linalg.norm(beam_vector) # include offset
            beam_vector_unit = beam_vector/parent_beam_length

            _tmp_len = 0.0
            for sub_beam_idx, sub_beam_length in enumerate(sub_beam_lengths):
                if sub_beam_length is None:
                    continue
                _tmp_len += sub_beam_length

            for sub_beam_idx, sub_beam_length in enumerate(sub_beam_lengths):
                if sub_beam_length is None:
                    sub_beam_lengths[sub_beam_idx] = parent_beam_length - _tmp_len

            # Create CON (CONE) sub-beams
            new_grup_info = []
            for sub_beam_idx, member_info in enumerate(grup_info):
                # sect_name, grup_dimensions, young_modulus, mass_density, sub_beam_length, flood_att, poisson_ratio\
                #     = member_info
                sect_label, grup_dimensions, young_modulus, mass_density, _, flood_att, poisson_ratio\
                    = member_info

                if sect_label in sect_infos:
                    sect_type, dimensions = sect_infos[sect_label]
                else:
                    sect_type = sect_label
                    dimensions = grup_dimensions

                if not sect_type == 'CON':
                    sub_grup_info = (sect_label, grup_dimensions, young_modulus, mass_density,
                                     sub_beam_lengths[sub_beam_idx], flood_att, poisson_ratio, None, None)
                    new_grup_info.append(sub_grup_info)
                    continue
                cone_sub_beam = 10
                sub_beam_len = sub_beam_lengths[sub_beam_idx] / cone_sub_beam

                if not sub_beam_idx == 0:
                    # If sub-group with CONE xsec (in GRUP line) is NOT located in the 1st line,
                    #  Joint A (with larger diameter) = node 0 (of sub-beam) => Use normal loop
                    for i in range(0, cone_sub_beam):
                        sect_type = 'TUB'
                        sub_dimensions = cone_diameter_interp(sub_beam_lengths[sub_beam_idx], sub_beam_len, dimensions, i)
                        sub_grup_info = (sect_label, grup_dimensions, young_modulus, mass_density,
                                         sub_beam_len, flood_att, poisson_ratio, sect_type, sub_dimensions)
                        new_grup_info.append(sub_grup_info)
                else:
                    # If sub-group with CONE xsec (in GRUP line) is located in the 1st line,
                    #  Joint A (with larger diameter) = node 0 (of sub-beam) => Use reverse loop
                    for i in range(cone_sub_beam, 0, -1):
                        sect_type = 'TUB'
                        sub_dimensions = cone_diameter_interp(sub_beam_lengths[sub_beam_idx], sub_beam_len, dimensions, i)
                        sub_grup_info = (sect_label, grup_dimensions, young_modulus, mass_density,
                                         sub_beam_len, flood_att, poisson_ratio, sect_type, sub_dimensions)
                        new_grup_info.append(sub_grup_info)
            grup_info = new_grup_info

            sub_beam_lengths = []
            for sub_beam_idx, member_info in enumerate(grup_info):
                _, _, _, _, sub_beam_length, _, _, _, _ = member_info
                sub_beam_lengths.append(sub_beam_length)

            sub_beam_coords_array = np.empty((len(grup_info), 2, 3))
            sub_beam_coords_array[:] = np.nan
            sub_beam_coords_array[0, 0, :] = offset_beam_coords[0]
            sub_beam_coords_array[-1, -1, :] = offset_beam_coords[-1]
            for sub_beam_idx, sub_beam_length in enumerate(sub_beam_lengths):
                if sub_beam_length is None:
                    break
                if np.isnan(sub_beam_coords_array[sub_beam_idx, 0, 0]):
                    break
                sub_beam_coords_array[sub_beam_idx, 1, :] = \
                    sub_beam_coords_array[sub_beam_idx, 0, :] + sub_beam_length*beam_vector_unit
                if sub_beam_idx < len(sub_beam_lengths) - 1:
                    sub_beam_coords_array[sub_beam_idx+1, 0, :] = sub_beam_coords_array[sub_beam_idx, 1, :]
            for sub_beam_idx, sub_beam_length in reversed(list(enumerate(sub_beam_lengths))):
                if sub_beam_length is None:
                    break
                if np.isnan(sub_beam_coords_array[sub_beam_idx, 1, 0]):
                    break
                sub_beam_coords_array[sub_beam_idx, 0, :] = \
                    sub_beam_coords_array[sub_beam_idx, 1, :] - sub_beam_length*beam_vector_unit
                if sub_beam_idx > 0:
                    sub_beam_coords_array[sub_beam_idx-1, 1, :] = sub_beam_coords_array[sub_beam_idx, 0, :]
            assert not np.any(np.isnan(sub_beam_coords_array)), sub_beam_coords_array

            beam_label_to_comp_id[joint_labels] = []
            for sub_beam_idx, member_info in enumerate(grup_info):
                sect_label, grup_dimensions, young_modulus, mass_density, _, \
                    flood_att_gr, poisson_ratio, sect_type, dimensions = member_info
                grup_ov_name, mgr_ov, flood_att_grp_ov, xsec_ov, dis_area_ov, cd_ov_gp, cm_ov_gp = grp_ov_info
                _, flood_att_mem_ov, _, cd_ov_m, cm_ov_m = mem_ov_info
                if xsec_ov == 0.001:
                    mass_density = 0.0
                if sect_type is None:
                    if sect_label in sect_infos:
                        sect_type, dimensions = sect_infos[sect_label]
                    else:
                        sect_type = sect_label
                        dimensions = grup_dimensions
                sub_beam_length = sub_beam_lengths[sub_beam_idx]
                sub_beam_coords = sub_beam_coords_array[sub_beam_idx]
                beam_label_to_comp_id[joint_labels].append((component_id, sub_beam_length))
                flood_att_original = attribute_override(flood_att_mem, flood_att_gr)
                flood_att_ov = attribute_override(flood_att_mem_ov, flood_att_grp_ov)
                flood_att = attribute_override(flood_att_ov, flood_att_original)
                cd_ov = attribute_override(cd_ov_m, cd_ov_gp)
                cm_ov = attribute_override(cm_ov_m, cm_ov_gp)

                beam, n_linear_springs = create_beam(
                    component_id, sub_beam_coords, beam_elem, joint_labels, sect_type, dimensions,
                    sect_label, rotation_angle, collection, young_modulus, mass_density,
                    cross_section_mesh_path, external_sect_lib, beam_label_to_comp_id, node_labels,
                    constraint_nodes, linear_spring_nodes, linear_springs, flood_att, poisson_ratio,
                    z_ref_joint_coord, sub_beam_idx=sub_beam_idx)
                component_id += 1 + n_linear_springs
                beams.append(beam)

                check_buoy, check_mgr, check_circular = _is_in_range(beam, mgr_ov_infos, seabed_elevation, water_elevation)
                marine_thickness, marine_density = _update_marine_growth(beam, mgr_ov_infos, mgr_ov, check_mgr)
                cd_normal, cm_normal = calculate_drag_mass_coef(beam, check_mgr, cdm_infos, check_circular)
                cd_normal = attribute_override(cd_ov, cd_normal)
                cm_normal = attribute_override(cm_ov, cm_normal)
                # THE BELOW FUNCTION IS ONLY TEMPORARILY DISABLED (DO NOT DELETE) !
                # _assign_linear_wave_load(
                #     beam, check_mgr, flood_att, water_density, seabed_elevation, water_elevation,  marine_thickness,
                #     marine_density, cd_normal, cm_normal, accel_0)
                beam_add_info = (
                check_buoy, check_mgr, flood_att, marine_thickness, marine_density, dis_area_ov, cd_normal, cm_normal,
                release_0, release_1)
                beam_add_infos.append(beam_add_info)

        progress_bar.increase()
    progress_bar.exit()

    # Create spring 6 dofs from beam offdet
    six_dofs_springs = []
    for spring_info in spring_6_dofs_infos:
        spring = rb.RBComponent(component_id, six_dofs_spring)
        spring_coords = np.array(spring_info)
        rotation_matrix, translation_vector, spring_len = \
            find_beam_transformation(spring_coords, is_spring=True)
        spring.rotation_matrix = rotation_matrix
        spring.translation_vector = translation_vector
        spring.mu_geo["length"].value = spring_len
        if IS_USE_RIGID is True:
            pass
            # spring.boolean_data["constrain_all_dofs"] = True
        else:
            spring.mu_coeff["spring_constant"].value = 1e+15
        spring.post_init()
        six_dofs_springs.append(spring)
        component_id += 1

    print (' => Adding', len(beams), 'beams,', len(linear_springs), 'linear springs and', \
        len(six_dofs_springs), 'six-dofs springs')
    # Well, after add components to the component_system, the component_id will change. See
    # component_system.add_components(), it's because we have builtin components so it re-orders...
    original_component_ids = component_system.add_components(
        shells+beams+linear_springs+six_dofs_springs, auto_connection=-1, signal=False,
        return_original_component_ids=True)
    component_id_map = np.argsort(original_component_ids)

    if len(all_collection_material_value_groups) > 0:
        # Write all used materials to collection.json
        ap.write_material_value_groups_to_collection_type(
            collection, all_collection_material_value_groups)

    # --------------------------------
    stored_selections = set()
    loads = set()
    load_cases = set()
    load_combinations = set()
    print ('> Create load case ...')
    isolated_node = collection.rb_ref_components['/builtin/builtin_isolated_node']
    all_point_load_node_idxs = {}
    mass_points = set()

    # Create buoyancy loadcase
    if CREATE_BUOYANCY_LOAD:
        operator_name = "linear_distributed_load"
        operator_type = operator_name_to_type[operator_name]

        # new_load_id = interaction_state.get_current_load_id()
        component_to_apply = beams
        additional_prop = beam_add_infos
        load_to_add_to_load_case = []

        for component, prop in zip(component_to_apply, additional_prop):
            check_buoy, _, _, _, _, disp_area_ov, _, _, _, _ = prop
            if check_buoy and disp_area_ov == None:
                diameter, thickness = API.get_diameter_from_port(component, 0)
                radius = diameter / 2.0
                length = component.get_parameter("length").value
                _, mgr_att, flood_att, marine_thickness, _, _, _, _, _, _ = prop

                force = get_buoyancy_force(mgr_att, flood_att, radius, thickness, length, marine_thickness,
                                           water_density, accel_0)
                parameters = {"coordinate_system": 0,
                              "force_a_x": 0.0, "force_a_y": 0.0, "force_a_z": force,
                              "force_b_x": 0.0, "force_b_y": 0.0, "force_b_z": force}
                load = ld.Load(len(loads), "Buoyancy for component {} - {}".format(component.get_id(), random_words()),
                               operator_type, parameters, [])

                for s in component.sources:
                    if s.get_operator_type() is operator_type:
                        load.add_source_instance(s)
                loads.add(load)
                load_to_add_to_load_case.append(load)

        if len(load_to_add_to_load_case) > 0:
            load_case = ld.LoadCase(len(load_cases), "Buoyancy Loads", load_to_add_to_load_case)
            load_cases.add(load_case)

    if CREATE_LOADS:
        # Create built-in Self-weight load case
        for suffix in ["SHELL", "STIFFENER", "BEAM"]:
            sw_loads = []
            sw_operator_type = operator_name_to_type[ot.SELF_WEIGHT]
            if suffix == "SHELL":
                components_to_apply = shells
            elif suffix == "BEAM":
                components_to_apply = beams
            value_groups = {}
            for component in components_to_apply:
                for s in component.sources:
                    if not s.get_operator_type() is sw_operator_type:
                        continue
                    test_geom_id = list(s.get_geom_ids())[0]
                    if suffix == "SHELL" and not test_geom_id.dimension == 2:
                        continue
                    if suffix == "STIFFENER" and not test_geom_id.dimension == 1:
                        continue
                    value_key = component.mu_coeff[s.parameters[0].name].value

                    if value_key not in value_groups:
                        value_groups[value_key] = set([s])
                    else:
                        value_groups[value_key].add(s)

            for value_key, source_groups in value_groups.items():
                # parameters = {"mass_density": value_key}
                parameters = {}
                _, value_key_dim, _ = dv.encode_param(
                    collection, None, value_key, "mass_density")
                value_key_dim = np.round(value_key_dim)
                load = ld.Load(len(loads), suffix+"-SELF-WEIGHT-"+str(value_key_dim),
                    sw_operator_type, parameters,  [])
                load.add_source_instances(source_groups, save_parameters=False, signal=False)
                loads.add(load)
                sw_loads.append(load)

            sw_load_case = ld.LoadCase(
                len(load_cases), suffix+"-SELF-WEIGHT", sw_loads)
            load_cases.add(sw_load_case)

        # Create load cases from Sacs data
        for load_case_info, load_case_content in load_infos.items():
            load_case_label, load_case_full_name = load_case_info
            load_to_add_to_load_case = []
            #print (">>", load_case_full_name
            debug = False
            if load_case_full_name == "":#""TOPSIDE OPEN AREA LL-JACKET INPLACE":
                debug = True
            for load_name, load_items in load_case_content.items():
                if debug:
                    print ("   >", load_name, len(load_items))
                value_groups = {}
                local_idx_key = 1
                for load_item, load_values in load_items.items():
                    load_type, item, load_direction = load_item
                    is_on_stiffener = False

                    if load_type == "NODE": # load on joint
                        value_key = (load_type, load_values, load_direction)
                        if value_key not in value_groups:
                            value_groups[value_key] = set([item])
                        else:
                            value_groups[value_key].add(item)
                    else: # load on beam
                        joint_labels = item
                        if joint_labels in beam_label_to_comp_id:
                            _beam_info = beam_label_to_comp_id[joint_labels]
                            if isinstance(_beam_info, tuple):
                                _, beam_length = beam_label_to_comp_id[joint_labels]
                            elif isinstance(_beam_info, list):
                                beam_length = [bi[1] for bi in _beam_info]
                            else:
                                assert False

                        elif joint_labels in stiffener_joint_labels:
                            _, _, beam_length = stiffener_joint_labels[joint_labels]
                            is_on_stiffener = True
                        else:
                            assert False
                            # if joint_labels not in all_beam_joint_labels:
                            #     continue

                        if load_type == "PART-UNIF": # Convert all nearly-full-beam partial load into linear load
                            load_length = load_values[8]
                            parent_beam_length = np.sum(beam_length)
                            if np.abs(parent_beam_length - load_length) <= 1e-3:
                                load_type = "UNIF"

                        if load_type == "PART-UNIF":
                            load_values = load_values + (beam_length,)

                        # no sub-beams
                        if not isinstance(beam_length, list):
                            if load_values[0] == "GLOB":
                                load_values = load_values + (is_on_stiffener, 0,)
                            else:
                                load_values = load_values + (is_on_stiffener, local_idx_key,)
                                local_idx_key += 1
                            value_key = (load_type, load_values, load_direction)
                            if value_key not in value_groups:
                                value_groups[value_key] = set([joint_labels])
                            else:
                                value_groups[value_key].add(joint_labels)
                        else:
                            # have sub-beams
                            if load_type in ["UNIF", "PART-UNIF"]:
                                parent_beam_length = np.sum(beam_length)
                                start_load_vector = np.array([load_values[1], load_values[2], load_values[3]])
                                end_load_vector = np.array([load_values[4], load_values[5], load_values[6]])
                                accumulated_length = 0.0
                                if load_type == "UNIF":
                                    load_position = (0., parent_beam_length)
                                    for sub_beam_idx in range(len(beam_length)):
                                        sub_beam_position = \
                                            (accumulated_length, accumulated_length+beam_length[sub_beam_idx])
                                        accumulated_length += beam_length[sub_beam_idx]
                                        x0, l, f0, f1 = calculate_sub_beam_linear_load(
                                            start_load_vector, end_load_vector,
                                            load_position, sub_beam_position)
                                        if x0 is None:
                                            continue
                                        sub_beam_load_values = \
                                            (load_values[0],
                                             f0[0], f0[1], f0[2],
                                             f1[0], f1[1], f1[2])
                                        if load_values[0] == "GLOB":
                                            load_values = sub_beam_load_values + \
                                                (is_on_stiffener, 0,)
                                        else:
                                            load_values = sub_beam_load_values + \
                                                (is_on_stiffener, local_idx_key,)
                                            local_idx_key += 1
                                        value_key = (load_type, load_values, load_direction)
                                        sub_beam_label = (joint_labels, sub_beam_idx)
                                        if value_key not in value_groups:
                                            value_groups[value_key] = set([sub_beam_label])
                                        else:
                                            value_groups[value_key].add(sub_beam_label)

                                elif load_type == "PART-UNIF":
                                    offset_0 = load_values[7]
                                    load_length = load_values[8]
                                    load_position = (offset_0, offset_0+load_length)
                                    for sub_beam_idx in range(len(beam_length)):
                                        sub_beam_position = \
                                            (accumulated_length, accumulated_length+beam_length[sub_beam_idx])
                                        accumulated_length += beam_length[sub_beam_idx]
                                        # if joint_labels == " 181 107" and debug:
                                        #     print (">>>>>>>> 181 107", sub_beam_idx
                                        #     print load_position, sub_beam_position
                                        x0, l, f0, f1 = calculate_sub_beam_linear_load(
                                            start_load_vector, end_load_vector,
                                            load_position, sub_beam_position)
                                        # if joint_labels == " 181 107" and debug:
                                        #     print ("--->"
                                        #     print x0, l, f0, f1
                                        if x0 is None:
                                            continue
                                        sub_beam_load_values = \
                                            (load_values[0],
                                             f0[0], f0[1], f0[2],
                                             f1[0], f1[1], f1[2], x0, l)
                                        if load_values[0] == "GLOB":
                                            load_values = sub_beam_load_values + \
                                                (beam_length[sub_beam_idx], is_on_stiffener, 0,)
                                        else:
                                            load_values = sub_beam_load_values + \
                                                (beam_length[sub_beam_idx], is_on_stiffener,
                                                 local_idx_key,)
                                            local_idx_key += 1
                                        value_key = (load_type, load_values, load_direction)
                                        sub_beam_label = (joint_labels, sub_beam_idx)
                                        if value_key not in value_groups:
                                            value_groups[value_key] = set([sub_beam_label])
                                        else:
                                            value_groups[value_key].add(sub_beam_label)
                                        # if joint_labels == " 181 107" and debug:
                                        #     print ("--->", sub_beam_label
                                        #     print value_key
                            elif load_type == "CONC":
                                load_position = load_values[1]
                                load_vector = np.array([load_values[2], load_values[3], load_values[4]])
                                accumulated_length = 0.0
                                for sub_beam_idx in range(len(beam_length)):
                                    sub_beam_position = \
                                        (accumulated_length, accumulated_length+beam_length[sub_beam_idx])
                                    accumulated_length += beam_length[sub_beam_idx]
                                    subbeam_offset = calculate_sub_beam_concentrated_load_position(
                                        load_position, sub_beam_position)
                                    if subbeam_offset is None:
                                        continue
                                    sub_beam_load_values = \
                                        (load_values[0],
                                         subbeam_offset, load_vector[0], load_vector[1], load_vector[2])
                                    if load_values[0] == "GLOB":
                                        load_values = sub_beam_load_values + (is_on_stiffener, 0,)
                                    else:
                                        load_values = sub_beam_load_values + (is_on_stiffener, local_idx_key,)
                                        local_idx_key += 1
                                    sub_beam_label = (joint_labels, sub_beam_idx)
                                    value_key = (load_type, load_values, load_direction)
                                    if value_key not in value_groups:
                                        value_groups[value_key] = set([sub_beam_label])
                                    else:
                                        value_groups[value_key].add(sub_beam_label)
                            else:
                                print("Haven't support load {} on sub-beam {}".format(
                                    load_type, joint_labels))
                                assert False

                subload_idx = 0
                for value_key, items in value_groups.items():
                    if debug:
                        print (value_key, items)
                    load_type, load_values, load_direction = value_key
                    components_to_apply = set()
                    load_represent_beam_labels = None
                    for item_idx, item in enumerate(items):
                        if load_type == "NODE":
                            joint_0 = item
                            node_idx = node_labels[joint_0]
                            node_coord = node_coords[node_idx]
                            if node_idx not in all_point_load_node_idxs:
                                # If not exists, create it
                                mass_point = rb.RBComponent(component_id, isolated_node)
                                mass_point.translation_vector = node_coord
                                mass_point.post_init()
                                mass_point.mu_geo["scale"].value = ISOLATED_SIZE
                                mass_point.set_tags(set([joint_0]))
                                mass_points.add(mass_point)
                                all_point_load_node_idxs[node_idx] = mass_point
                                component_id += 1
                            else:
                                mass_point = all_point_load_node_idxs[node_idx]
                            components_to_apply.add(mass_point)

                        elif load_type in ["UNIF", "PART-UNIF", "CONC"]:
                            is_on_stiffener = load_values[-2]
                            if not is_on_stiffener:
                                if not isinstance(item, tuple):
                                    joint_labels = item
                                    load_represent_beam_labels = joint_labels
                                    _beam_comp_id, _ = beam_label_to_comp_id[joint_labels]
                                    # _beam_comp_id is component_id before adding to
                                    # component_system; after adding to component_system, it changes
                                    beam_comp_id = component_id_map[_beam_comp_id]
                                    beam = component_system.components[beam_comp_id]
                                    components_to_apply.add(beam)
                                    # if debug:
                                    #     print (" +beam_comp_id", beam_comp_id, joint_labels
                                else:
                                    assert len(item) == 2
                                    joint_labels, sub_beam_idx = item
                                    _beam_info = beam_label_to_comp_id[joint_labels]
                                    for _sub_beam_idx, bi in enumerate(_beam_info):
                                        if not _sub_beam_idx == sub_beam_idx:
                                            continue
                                        _beam_comp_id, _ = bi
                                        # _beam_comp_id is component_id before adding to
                                        # component_system; after adding to component_system, it changes
                                        beam_comp_id = component_id_map[_beam_comp_id]
                                        beam = component_system.components[beam_comp_id]
                                        components_to_apply.add(beam)
                                    #     if debug:
                                    #         print (" + sub beam_comp_id", beam_comp_id, joint_labels
                                    # if joint_labels == " 181 107" and debug:
                                    #     print ("---> FOUND", item
                                    #     print   'add sub-beam ID', beam_comp_id, load_values
                                    load_represent_beam_labels = joint_labels + "#" + str(sub_beam_idx)
                            else:
                                try:
                                    shell_comp_id, stiffener_block_id, _ = \
                                        stiffener_joint_labels[joint_labels]
                                except KeyError:
                                    # TODO: Trong check meSHing script
                                    print("Cannot apply load find stiffener block for member", joint_labels)
                                    continue
                                shell = component_system.components[shell_comp_id]
                                components_to_apply.add(shell)
                        else:
                            assert False

                    if len(components_to_apply) == 0:
                        continue

                    if load_type == "NODE":
                        operator_type = operator_name_to_type["point_load"]
                        parameters = {
                            "point_load_x": load_values[0],
                            "point_load_y": load_values[1],
                            "point_load_z": load_values[2],
                            "point_moment_x": load_values[3],
                            "point_moment_y": load_values[4],
                            "point_moment_z": load_values[5]
                        }
                    elif load_type == "UNIF":
                        operator_name = "linear_distributed_load"
                        if is_on_stiffener:
                            operator_name = "stiffener_" + operator_name
                        operator_type = operator_name_to_type[operator_name]
                        force_a = np.array([load_values[1], load_values[2], load_values[3]])
                        force_b = np.array([load_values[4], load_values[5], load_values[6]])
                        if load_values[0] == "MEMB":
                            assert len(components_to_apply) == 1
                            _beam = list(components_to_apply)[0]
                            if _beam.is_predefined_beam():
                                rotation_matrix = _beam.rotation_matrix
                                predefined_beam_name = _beam.get_name()
                                force_a = transform_local_force_on_beam(
                                    predefined_beam_name, rotation_matrix, force_a)
                                force_b = transform_local_force_on_beam(
                                    predefined_beam_name, rotation_matrix, force_b)
                                #print ("after", force_a
                        parameters = {
                            "coordinate_system": get_coordinate_system_idx(load_values[0]),
                            "force_a_x": force_a[0],
                            "force_a_y": force_a[1],
                            "force_a_z": force_a[2],
                            "force_b_x": force_b[0],
                            "force_b_y": force_b[1],
                            "force_b_z": force_b[2]
                        }
                    elif load_type == "PART-UNIF":
                        beam_length = load_values[-3]
                        load_length = load_values[-4]
                        offset_0 = load_values[-5]

                        if isinstance(beam_length, tuple):
                            print (load_length, beam_length, load_length > beam_length)
                            assert False

                        if load_length < 1e-6:
                            load_length = beam_length - offset_0

                        if load_length > beam_length:
                            if (load_length - (beam_length - offset_0)) > 0.01:
                                print ("WARNING: {}, load name {} has load_length = {} > beam_length = {}".format(
                                    joint_labels, load_name, load_length, beam_length))
                                assert False
                            else:
                                load_length = beam_length - offset_0

                        if load_length <= 0.:
                            print ("ERROR:", load_values, is_on_stiffener)
                            print (offset_0, beam_length, load_length, joint_labels)
                            continue
                            # assert load_length > 0.

                        operator_name = "partial_distributed_load"
                        if is_on_stiffener:
                            operator_name = "stiffener_" + operator_name
                        operator_type = operator_name_to_type[operator_name]
                        force_a = np.array([load_values[1], load_values[2], load_values[3]])
                        force_b = np.array([load_values[4], load_values[5], load_values[6]])

                        if load_values[0] == "MEMB":
                            assert len(components_to_apply) == 1
                            _beam = list(components_to_apply)[0]
                            if _beam.is_predefined_beam():
                                if debug:
                                    print (force_a)
                                rotation_matrix = _beam.rotation_matrix
                                predefined_beam_name = _beam.get_name()
                                force_a = transform_local_force_on_beam(
                                    predefined_beam_name, rotation_matrix, force_a, debug=debug)
                                force_b = transform_local_force_on_beam(
                                    predefined_beam_name, rotation_matrix, force_b)
                                if debug:
                                    print (force_a)
                        # if debug:
                        #     print ("CDS:", load_values[0], get_coordinate_system_idx(load_values[0])
                        parameters = {
                            "coordinate_system_partial_distributed_load": get_coordinate_system_idx(load_values[0]),
                            "force_1_x": force_a[0],
                            "force_1_y": force_a[1],
                            "force_1_z": force_a[2],
                            "force_2_x": force_b[0],
                            "force_2_y": force_b[1],
                            "force_2_z": force_b[2],
                            "distance_from_node_a": offset_0,
                            "load_length": load_length
                        }
                    elif load_type == "CONC":
                        operator_name = "concentrated_load"
                        if is_on_stiffener:
                            operator_name = "stiffener_" + operator_name
                        operator_type = operator_name_to_type[operator_name]
                        parameters = {
                            "coordinate_system_concentrated_load": get_coordinate_system_idx(load_values[0]),
                            "distance_from_node_0": load_values[1],
                            "force_x": load_values[2],
                            "force_y": load_values[3],
                            "force_z": load_values[4]
                        }
                    else:
                        assert False

                    if len(value_groups) == 1:
                        load = ld.Load(len(loads), load_name, operator_type, parameters, [])
                    else:
                        sub_load_name = load_name
                        if isinstance(load_direction, str):
                            sub_load_name += '|' + load_direction
                        if load_represent_beam_labels is not None:
                            sub_load_name += '|' + load_represent_beam_labels
                        else:
                            sub_load_name += "|(" + str(subload_idx)+")"
                            subload_idx += 1
                        if len(components_to_apply) > 1:
                            sub_load_name += "(group)"
                        load = ld.Load(
                            len(loads), sub_load_name, operator_type, parameters,  [])

                    if debug:
                        print ("components_to_apply", [c.get_id() for c in components_to_apply])
                    for component in components_to_apply:
                        for s in component.sources:
                            if not s.get_operator_type() is operator_type:
                                continue
                            load.add_source_instance(s)
                    # if load_represent_beam_labels is not None:
                    #     if load_represent_beam_labels != " 181 107":
                    #         continue
                    loads.add(load)
                    load_to_add_to_load_case.append(load)

            if len(load_to_add_to_load_case) > 0:
                load_case = ld.LoadCase(
                    len(load_cases), load_case_label + '-' + load_case_full_name,
                    load_to_add_to_load_case)
                load_cases.add(load_case)

        component_system.add_components(list(mass_points), auto_connection=-1, signal=False)

    print ('> Auto-connecting components ...')
    component_system.refresh_list(1)

    # Invalid connector is connector that only connects to one component (it should connect to two).
    # Another case is when two connectors connect together and both connect to the same component.
    count = 0
    invalid_connector_ids = check_invalid_connector(component_system)
    while len(invalid_connector_ids) > 0:
        count = count + len(invalid_connector_ids)
        invalid_connectors = [component_system.components[id] for id in invalid_connector_ids]
        component_system.remove_components(invalid_connectors, signal=False)
        invalid_connector_ids = check_invalid_connector(component_system)
    print (" > Remove {} invalid spring/rigid connectors".format(count))

    print ('> Create load combination ...')
    for load_combination_name, load_combination_info in load_combination_infos.items():
        load_combination = ld.LoadCombination(len(load_combinations), load_combination_name, [])
        for load_id, load_coefficient in load_combination_info:
            load_case = find_load_case_by_sacs_load_id(load_id, load_cases)
            if load_case is None:
                print ("INFO: Could not find load case ID", load_id)
                continue
            load_combination.add_load_case_contribution(load_case, load_coefficient)
        load_combinations.add(load_combination)

    load_datas = []
    for load in loads:
        load_datas.append(load.pre_write())
    load_case_datas = []
    for load_case in load_cases:
        load_case_datas.append(load_case.pre_write(list(loads)))
    load_combination_datas = []
    for load_combination in load_combinations:
        load_combination_datas.append(load_combination.pre_write(list(load_cases)))

    component_system_data = component_system.pre_write()
    stored_selection_data = []
    for stored_selection in stored_selections:
        stored_selection_data.append(stored_selection.pre_write())
    akselos_assembly = aa.AkselosAssembly(
        component_system=component_system_data,
        stored_selections=stored_selection_data,
        load_cases=load_datas,
        multi_load_cases=load_case_datas,
        load_combinations=load_combination_datas,
        current_load_combination=0)
    aks_dir = os.path.join(working_collection_path, 'aks_files')
    if not os.path.exists(aks_dir):
        os.mkdir(aks_dir)
        aks_filepath = os.path.join(aks_dir, 'model.aks')

    print ("> Writing data to AKS file:", aks_filepath)
    fid = open(aks_filepath, 'w')
    fid.write(json_helper.write_json_str(akselos_assembly, unformatted_fast=False))
    fid.close()

    print ("> Writing LIC plugin data")
    _assign_LIC_plugin(
        aks_filepath, beams, beam_add_infos, water_density, seabed_elevation, water_elevation, accel_0)
    _assign_release_att(aks_filepath, beams, beam_add_infos)
    modify_master_port(aks_filepath, beams, beam_add_infos)


