import os
import weakref
import tempfile

import numpy as np

import face_ref as fr
import handler as hn
import id_objects
import mesh as ms
import qt
import selector
import selection_mode as sm
import ui_objects
import utils
import graphics.graphics_api as ga
import operator_id as oi
import operator_type as ot
import graphics_objects.widgets as ws
import graphics_objects.graphics_body as gb
import point as pt
import interfaces.types as tp
import new_type
import akselos.mesh as am
import dimensional_value as dv
import user_error_message
import types_data

import id_objects.stiffener_id as si

from . import port_type_ui
from . import ref_component_type_graphics
from . import ref_component_type_ui as rf


class RefComponentTypeGeomUI:
    def __init__(self, ref_component_type_geom_id):
        self.weak_ref_component_type_geom_id = weakref.ref(ref_component_type_geom_id)

    def __getattr__(self, name):
        if name == 'ref_component_type_geom_id':
            return self.weak_ref_component_type_geom_id()

        raise AttributeError(name)

    @staticmethod
    @utils.memoized(weak_key=True)
    def get(ref_component_type_geom_id):
        assert isinstance(ref_component_type_geom_id, id_objects.RefTypeGeomId)
        return RefComponentTypeGeomUI(ref_component_type_geom_id)

    @staticmethod
    def create_new_surface(updater, ref_component_type, side_idxs=None):
        # Modify the ref_component_type's mesh by adding a new surface with (optional) side_idxs.
        # Returns the surface_id.
        if side_idxs is None:
            side_idxs = []
        mesh = ref_component_type.get_default_mesh()
        used = list(mesh.labeled_surfaces.keys())
        if len(used) > 0:
            labeled_surface_mesh_ids = [geom_id.mesh_id for geom_id in used]
            max_mesh_id = max(labeled_surface_mesh_ids)
            new_surface_id = ms.geom_id.GeomId(max_mesh_id + 1, 2, False)
        else:
            new_surface_id = ms.geom_id.GeomId(1, 2, False)

        mesh.set_sideset(new_surface_id, side_idxs)
        ref_component_type_ui = rf.RefComponentTypeUI.get(ref_component_type)
        ref_component_type_ui.apply_changes(updater, mesh_modified=True)
        ref_component_type_surface_id = id_objects.RefTypeGeomId(
            ref_component_type, new_surface_id.mesh_id, 2, is_subdomain=False)
        updater.update_tree_after_new_item_is_added(
            ref_component_type, ref_component_type_surface_id)
        updater.ref_component_mesh_changed(ref_component_type)

        return new_surface_id

    @staticmethod
    def create_new_nodeset(updater, ref_component_type, node_idxs=None):
        # Modify the ref_component_type's mesh by adding a new nodeset.
        if node_idxs is None:
            node_idxs = []
        mesh = ref_component_type.get_default_mesh()
        used = list(mesh.labeled_surfaces.keys()) + list(mesh.labeled_vertices.keys())
        if len(used) > 0:
            labeled_mesh_ids = [geom_id.mesh_id for geom_id in used]
            max_mesh_id = max(labeled_mesh_ids)
            new_nodeset_id\
                = ms.geom_id.GeomId(max_mesh_id + 1, 0, False)
        else:
            new_nodeset_id = ms.geom_id.GeomId(1, 0, False)

        mesh.set_nodeset(new_nodeset_id, node_idxs)
        ref_component_type_ui = rf.RefComponentTypeUI.get(ref_component_type)
        ref_component_type_ui.apply_changes(updater, mesh_modified=True)
        ref_component_type_nodeset_id = id_objects.RefTypeGeomId(
            ref_component_type, new_nodeset_id.mesh_id, 0, is_subdomain=False)
        updater.update_tree_after_new_item_is_added(
            ref_component_type, ref_component_type_nodeset_id)
        updater.ref_component_mesh_changed(ref_component_type)

        return ref_component_type_nodeset_id

    @staticmethod
    def remove_geometry(updater, ref_component_type, geom_id):
        # Modify the ref_component_type's mesh by removing a geometry
        mesh = ref_component_type.get_default_mesh()
        success = mesh.remove_geometry(geom_id)
        if success:
            ref_component_type_ui = rf.RefComponentTypeUI.get(ref_component_type)
            ref_component_type_ui.apply_changes(updater, mesh_modified=True)
            parent_node = updater.main.structure_tree.structure_model\
                .get_node_from_item(ref_component_type)
            updater.update_tree_after_item_is_deleted(parent_node)
            updater.ref_component_mesh_changed(ref_component_type)
        return success

    @staticmethod
    def auto_assign_edge_port_to_port_type_geom_id(port_type_geom_id, use_non_conforming,
                                                   relative_tol, allow_scaling):
        geom_id = port_type_geom_id.get_single_geom_id()
        if geom_id is None:
            return False, "Can only assign a port to a single mesh id."
        port_type = port_type_geom_id.port_type
        collection = port_type.collection_type
        found_port_type = False
        edge_port_types = collection.get_edge_port_types()

        for edge_port_type in edge_port_types:
            success, error_message = ui_objects.RefTypePortUI.apply_edge_port_to_port_type_geom_id(
                port_type_geom_id, edge_port_type, use_non_conforming, relative_tol, allow_scaling)
            if success:
                found_port_type = True
                message = 'Assigned ' + str(edge_port_type.core.name) + ' to geom id ' + str(
                        geom_id.mesh_id)
                break
        if not found_port_type:
            message = 'No valid port types found for geom ' + str([geom_id.mesh_id])

        return found_port_type, message

    @staticmethod
    def auto_assign_port(ref_type_geom_id, use_non_conforming=False, relative_tol=1e-3,
                         allow_scaling=False, user_input_port_mesh_ids=None,
                         add_free_nodeset_to_mesh=False):
        ref_component_type = ref_type_geom_id.ref_component_type
        geom_id = ref_type_geom_id.get_single_geom_id()
        if geom_id is None:
            return False, "Can only assign a port to a single mesh id."

        collection = ref_component_type.collection_type
        success = False
        message = None
        port_dimension = ref_type_geom_id.get_port_dimension()
        if port_dimension == 0:
            port_types = [collection.rb_component_node_dof_object]
        elif port_dimension == 1:
            port_types = collection.get_edge_port_types()
        elif port_dimension == 2:
            port_types = collection.get_port_types()
        else:
            return False, "Could not match port of dimension " + str(port_dimension)

        # mesh = ref_component_type.get_default_mesh()
        # updater = ref_type_geom_id.canvas.istate.main.updater
        #
        # for edge_id in sorted(mesh.get_geom_ids(dimension=1, is_subdomain=False)):
        #     # if add_extra_point_to_edge_port:
        #     if True:
        #         # joint_with_stiffener, extra_stiffener_point  = RefComponentTypeGeomUI\
        #         #     .check_edge_and_stiffener_intersection(mesh, edge_id, stiffener_infos)
        #         joint_with_stiffener, extra_stiffener_point = RefComponentTypeGeomUI \
        #             .check_edge_and_stiffener_intersection_2(mesh, edge_id)
        #     # else:
        #     #     joint_with_stiffener = False
        #     #     extra_stiffener_point = None
        #
        #     RefComponentTypeGeomUI.create_new_face_port_type(
        #         ref_type_geom_id, mesh, collection, edge_id, ref_component_type,
        #         updater, use_non_conforming, relative_tol, allow_scaling, user_input_port_mesh_ids, summary_box = None
        #         , add_free_nodeset_to_mesh=add_free_nodeset_to_mesh, extra_stiffener_point=extra_stiffener_point)

        for port_type in port_types:
            try:
                if port_dimension != 0:
                    port_type.get_default_mesh()
            except RuntimeError:
                # Ugh, this happens if we don't have the mesh file.  Normally, this is an
                # "oh well" sort of situation, but here we don't want this whole loop to die
                # because of one invalid port.
                continue

            # Trong: sometimes we do not want to re-use port widely, we just need one port_type at its original location
            # for 2 components. Using the same port_type for several positions may lead to "port mesh mismatch"
            # For now, this is not the default option, production team can enable it manually when needed.
            # ----
            # center_1 = port_type.get_default_mesh().mapped_mesh.node_cen[:3]
            # center_2 = ref_type_geom_id.get_model_center_point()
            # if not(np.linalg.norm(center_1-center_2) < 0.001):
            #     continue
            # ----

            success, message = ui_objects.RefTypePortUI.apply_port_to_ref_type_geom_id(
                ref_type_geom_id, port_type, use_non_conforming, relative_tol, allow_scaling,
                user_input_port_mesh_ids=user_input_port_mesh_ids,
                add_free_nodeset_to_mesh=add_free_nodeset_to_mesh)

            # TODO: Here we are stopping at the first matching port type. This way will reduce the
            # running time. But the optimal way is to iterate all the port types, find all the
            # matching port types and choose the one having smallest error.
            if success:
                if port_dimension == 0:
                    message = 'Assigned ' + str(port_type.core.name) + ' to nodeset ' + str(
                        [geom_id.mesh_id])
                elif port_dimension in [1, 2]:
                    if geom_id.is_subdomain and geom_id.dimension == 1:
                        message = 'Assigned ' + str(port_type.core.name) + ' to edge block ' + str(
                            [geom_id.mesh_id])
                    else:
                        message = 'Assigned ' + str(port_type.core.name) + ' to sideset ' + str(
                            [geom_id.mesh_id])
                break

        return success, message

    @staticmethod
    def auto_assign_ports(ref_component_type, use_non_conforming=False, relative_tol=1e-3,
                          allow_scaling=False, ids_rules=None, summary_box=None):
        if ids_rules is None:
            ids_rules = {"min": 100, "max": 199}
        mesh = ref_component_type.get_default_mesh()
        any_success = False
        for mesh_dimension in [1, 2]:
            for geom_id in sorted(mesh.get_geom_ids(dimension=mesh_dimension)):
                if geom_id.mesh_id < ids_rules["min"] or geom_id.mesh_id > ids_rules["max"]:
                    continue

                ref_type_geom_id = id_objects.RefTypeGeomId(
                    ref_component_type, geom_id.mesh_id, mesh_dimension, is_subdomain=False)
                port_idx = ref_type_geom_id.get_port_idx()
                if port_idx is not None:
                    continue

                success, message = RefComponentTypeGeomUI.auto_assign_port(
                    ref_type_geom_id, use_non_conforming, relative_tol, allow_scaling)

                if summary_box is not None:
                    if success:
                        summary_box.add_info(message)
                    else:
                        summary_box.add_error(
                            'No valid port type found for sideset' + str([geom_id.mesh_id]))
                any_success |= success
        return any_success

    @staticmethod
    def assign_sidset_as_source(ref_component_type, operator_type, geom_ids, geom_dim,
                                geom_name, summary_box):
        ref_component_type.c_add_geometry(geom_dim, geom_ids, str(geom_name))
        new_source_type = ref_component_type.add_new_source(
            operator_type, source_gui_name=str(geom_name))
        operator_id = oi.SourceOperatorId(ref_component_type, new_source_type.core.name)
        mesh_ids = [geom_id.mesh_id for geom_id in geom_ids]
        operator_id.set_mesh_ids(mesh_ids)
        summary_box.add_info(
            "Assigning " + str(operator_type.name) + " to sideset " + str(mesh_ids))

    @staticmethod
    def assign_sideset_as_dirichlet(ref_component_type, geom_id, geom_dim, geom_name, summary_box):
        success = ref_component_type.add_new_dirichlet_geom(geom_id, geom_dim, geom_name=geom_name)
        if success:
            summary_box.add_info(
                "Assigning dirichlet_boundary_conditions (x=0; y=0; z=0)" + " to sideset " + str(
                    [geom_id.mesh_id]))

    @staticmethod
    def assign_nodeset_as_node_port(ref_component_type, nodeset_id, node_idxs, summary_box=None):
        ref_component_type.add_single_node_port(nodeset_id, node_idxs)
        summary_box.add_info("Assigning node_port to nodeset " + str([nodeset_id.mesh_id]))

    @staticmethod
    def assign_nodeset_as_point_load(ref_component_type, nodeset_id, summary_box=None):
        ref_component_type.c_add_geometry(0, [nodeset_id])
        physics_type = ref_component_type.collection_type.get_physics_type()
        operator_name_to_type = physics_type.get_operator_name_to_type()
        operator_type = operator_name_to_type[ot.POINT_LOAD]
        new_source_type = ref_component_type.add_new_source(operator_type)
        operator_id = oi.SourceOperatorId(ref_component_type, new_source_type.core.name)
        operator_id.set_mesh_ids({nodeset_id.mesh_id})
        summary_box.add_info(
            "Assigning " + str(operator_type.name) + " to nodeset " + str([nodeset_id.mesh_id]))

    @staticmethod
    def assign_sideset_as_spring_surface(ref_component_type, geom_id, geom_dim, geom_name,
                                         summary_box, direction, is_shell):
        assert geom_dim == 2, "Can only assign spring to a surface"
        collection_type = ref_component_type.collection_type
        ref_component_type.add_new_spring_surface(
            geom_id, geom_name, direction, collection_type, is_shell)
        summary_box.add_info("Assigning directional spring to sideset " + str([geom_id.mesh_id]))

    @staticmethod
    def assign_sideset_as_contact_surface(ref_component_type, geom_id, geom_dim, geom_name,
                                         summary_box):
        assert geom_dim == 2, "Can only assign contact to a surface"
        ref_component_type.add_contact_surface(geom_id, geom_name)
        summary_box.add_info("Assigning contact surface to sideset " + str([geom_id.mesh_id]))

    @staticmethod
    def auto_assign_sideset_using_id_convention(ref_component_type, use_non_conforming,
                                                relative_tol, allow_scaling,summary_box,
                                                id_parameters, updater, extract_missing_port,
                                                add_free_nodeset_to_mesh,
                                                add_extra_point_to_edge_port):
        port_ids_range = id_parameters.port_ids_range
        normal_load_ids_range = id_parameters.normalload_ids_range
        surface_load_ids_range = id_parameters.surfaceload_ids_range
        dirichlet_ids_range = id_parameters.dirichlet_ids_range
        hydrostatic_ids_range = id_parameters.hydrostatic_ids_range
        spring_x_ids_range = id_parameters.spring_x_ids_range
        spring_y_ids_range = id_parameters.spring_y_ids_range
        spring_z_ids_range = id_parameters.spring_z_ids_range
        contact_range = id_parameters.contact_range
        mesh = ref_component_type.get_default_mesh()

        sideset_names = mesh.mesh_data.sideset_names
        if sideset_names is None:
            sideset_names = {}

        collection_type = ref_component_type.collection_type
        physics_type = collection_type.get_physics_type()
        operator_name_to_type = physics_type.get_operator_name_to_type()
        user_input_port_mesh_ids = tuple(np.arange(port_ids_range['min'], port_ids_range['max']+1))
        # Get all the hydrostatic load surfaces first then at the end assign all of them as
        # single source.
        hydrodtatic_surface_ids = []

        # sideset that is surface
        for surface_id in sorted(mesh.get_geom_ids(dimension=2, is_subdomain=False)):
            surface_name = sideset_names.get(
                surface_id.mesh_id, 'Surface_' + str(surface_id.mesh_id))

            # TODO: In the long term, we should write a function to defect if a given surface lie on
            # solid or shell blocks or both (does it already exist?)
            # For now, we decide based on the modeling type:
            is_shell_surface = ref_component_type.is_shell()
            is_mixed, _, _, solid_block_ids = mesh.check_mixed_elememts()
            if is_mixed:
                # For now, if modeling type is "mixed" and has no solid blocks then assume that
                # surface should be shell.
                is_shell_surface |= len(solid_block_ids) == 0

            # Port assignment
            if port_ids_range["min"] <= surface_id.mesh_id <= port_ids_range["max"]:
                ref_type_geom_id = id_objects.RefTypeGeomId(
                    ref_component_type, surface_id.mesh_id, surface_id.dimension,
                    is_subdomain=False)
                port_idx = ref_type_geom_id.get_port_idx()
                if port_idx is not None:
                    continue
                success, message = RefComponentTypeGeomUI.auto_assign_port(
                    ref_type_geom_id, use_non_conforming, relative_tol, allow_scaling,
                    user_input_port_mesh_ids, add_free_nodeset_to_mesh)
                if success:
                    if summary_box is not None:
                        summary_box.add_info(message)
                elif extract_missing_port:
                    # If there is any sideset that users want to become port (i.e. has ID in the
                    # port ID range) but it cannot (because there is no appropriate port type),
                    # then here the Editor will extract mesh from that sideset make new port type.
                    # And then assign the new port type back to the sideset.
                    summary_box.add_info(
                        'No valid port type found for sideset ' + str([surface_id.mesh_id]),
                        color='red')
                    RefComponentTypeGeomUI.create_new_face_port_type(
                        ref_type_geom_id, mesh, collection_type, surface_id, ref_component_type,
                        updater, use_non_conforming, relative_tol, allow_scaling,
                        user_input_port_mesh_ids, summary_box, add_free_nodeset_to_mesh)
                else:
                    if summary_box is not None:
                        summary_box.add_error(
                            'No valid port type found for sideset ' + str([surface_id.mesh_id]))
            # Normal load assignment
            elif normal_load_ids_range["min"] <= surface_id.mesh_id <= normal_load_ids_range["max"]:
                if ot.NORMAL_LOAD not in operator_name_to_type \
                        or ot.SHELL_NORMAL_LOAD not in operator_name_to_type:
                    continue
                if is_shell_surface:
                    operator_type = operator_name_to_type[ot.SHELL_NORMAL_LOAD]
                else:
                    operator_type = operator_name_to_type[ot.NORMAL_LOAD]
                RefComponentTypeGeomUI.assign_sidset_as_source(
                    ref_component_type, operator_type, [surface_id], 2, surface_name, summary_box)
            # Surface load assignment
            elif surface_load_ids_range["min"] <= surface_id.mesh_id <= surface_load_ids_range["max"]:
                if ot.SURFACE_LOAD not in operator_name_to_type \
                        or ot.SHELL_SURFACE_LOAD not in operator_name_to_type:
                    continue
                if is_shell_surface:
                    operator_type = operator_name_to_type[ot.SHELL_SURFACE_LOAD]
                else:
                    operator_type = operator_name_to_type[ot.SURFACE_LOAD]
                RefComponentTypeGeomUI.assign_sidset_as_source(
                    ref_component_type, operator_type, [surface_id], 2, surface_name, summary_box)
            # Dirichlet boundary assignment
            elif dirichlet_ids_range["min"] <= surface_id.mesh_id <= dirichlet_ids_range["max"]:
                RefComponentTypeGeomUI.assign_sideset_as_dirichlet(
                    ref_component_type, surface_id, 2, surface_name, summary_box)
            # Hydrostatic assignment
            if hydrostatic_ids_range["min"] <= surface_id.mesh_id <= hydrostatic_ids_range["max"]:
                hydrodtatic_surface_ids.append(surface_id)

            # Spring surface assignment, can be assign along side with other surface load
            if spring_x_ids_range["min"] <= surface_id.mesh_id <= spring_x_ids_range["max"]:
                direction = "x"
                RefComponentTypeGeomUI.assign_sideset_as_spring_surface(
                    ref_component_type, surface_id, 2, surface_name, summary_box, direction,
                    is_shell_surface)

            if spring_y_ids_range["min"] <= surface_id.mesh_id <= spring_y_ids_range["max"]:
                direction = "y"
                RefComponentTypeGeomUI.assign_sideset_as_spring_surface(
                    ref_component_type, surface_id, 2, surface_name, summary_box, direction,
                    is_shell_surface)

            if spring_z_ids_range["min"] <= surface_id.mesh_id <= spring_z_ids_range["max"]:
                direction = "z"
                RefComponentTypeGeomUI.assign_sideset_as_spring_surface(
                    ref_component_type, surface_id, 2, surface_name, summary_box, direction,
                    is_shell_surface)

            if contact_range["min"] <= surface_id.mesh_id <= contact_range["max"]:
                RefComponentTypeGeomUI.assign_sideset_as_contact_surface(
                    ref_component_type, surface_id, 2, surface_name, summary_box)
            ref_component_type.clear_method_cache()

        # if add_extra_point_to_edge_port:
        #     stiffener_infos = si.StiffenerId.create_for_ref_component_type(ref_component_type)

        # shell sideset that is edge
        for edge_id in sorted(mesh.get_geom_ids(dimension=1, is_subdomain=False)):
            edge_name = sideset_names.get(edge_id.mesh_id, 'Edge_' + str(edge_id.mesh_id))
            if port_ids_range["min"] <= edge_id.mesh_id <= port_ids_range["max"]:
                if add_extra_point_to_edge_port:
                    # joint_with_stiffener, extra_stiffener_point  = RefComponentTypeGeomUI\
                    #     .check_edge_and_stiffener_intersection(mesh, edge_id, stiffener_infos)
                    joint_with_stiffener, extra_stiffener_point = RefComponentTypeGeomUI \
                        .check_edge_and_stiffener_intersection_2(mesh, edge_id)
                else:
                    joint_with_stiffener = False
                    extra_stiffener_point = None

                ref_type_geom_id = id_objects.RefTypeGeomId(
                    ref_component_type, edge_id.mesh_id, edge_id.dimension, edge_id.is_subdomain, joint_with_stiffener)
                port_idx = ref_type_geom_id.get_port_idx()
                if port_idx is not None:
                    continue
                success, message = RefComponentTypeGeomUI.auto_assign_port(
                    ref_type_geom_id, use_non_conforming, relative_tol,
                    allow_scaling, user_input_port_mesh_ids)
                if success:
                    if summary_box is not None:
                        summary_box.add_info(message)
                elif extract_missing_port:
                    if summary_box is not None:
                        summary_box.add_info(
                            'No valid port type found for sideset '+str([edge_id.mesh_id]), color='red')
                    RefComponentTypeGeomUI.create_new_face_port_type(
                        ref_type_geom_id, mesh, collection_type, edge_id, ref_component_type,
                        updater, use_non_conforming, relative_tol, allow_scaling, user_input_port_mesh_ids,
                        summary_box, add_free_nodeset_to_mesh, extra_stiffener_point=extra_stiffener_point)
                else:
                    if summary_box is not None:
                        summary_box.add_error(
                            'No valid port type found for sideset '+ str([edge_id.mesh_id]))
            elif normal_load_ids_range["min"] <= edge_id.mesh_id <= normal_load_ids_range["max"]:
                continue
            elif surface_load_ids_range["min"] <= edge_id.mesh_id <= surface_load_ids_range["max"]:
                continue
            elif dirichlet_ids_range["min"] <= edge_id.mesh_id <= dirichlet_ids_range["max"]:
                RefComponentTypeGeomUI.assign_sideset_as_dirichlet(
                    ref_component_type, edge_id, 1, edge_name, summary_box)
            elif hydrostatic_ids_range["min"] <= edge_id.mesh_id <= hydrostatic_ids_range["max"]:
                continue
            elif spring_x_ids_range["min"] <= edge_id.mesh_id <= spring_x_ids_range["max"]:
                continue
            elif spring_y_ids_range["min"] <= edge_id.mesh_id <= spring_y_ids_range["max"]:
                continue
            elif spring_z_ids_range["min"] <= edge_id.mesh_id <= spring_z_ids_range["max"]:
                continue
            ref_component_type.clear_method_cache()

        # Assign all hydrostatic surfaces into a single source load
        if ref_component_type.is_shell():
            operator_type = operator_name_to_type.get(ot.SHELL_HYDROSTATIC_LOAD, None)
        else:
            operator_type = operator_name_to_type.get(ot.HYDROSTATIC_LOAD, None)
        if operator_type is not None and len(hydrodtatic_surface_ids) > 0:
            # surface_name = "Hydrostatic_surfaces"
            # RefComponentTypeGeomUI.assign_sidset_as_source(
            #     ref_component_type, operator_type, hydrodtatic_surface_ids,
            #     2, surface_name, summary_box)
            for hydrodtatic_surface_id in hydrodtatic_surface_ids:
                surface_name = "Hydrostatic_surfaces"
                RefComponentTypeGeomUI.assign_sidset_as_source(
                    ref_component_type, operator_type, [hydrodtatic_surface_id],
                    2, surface_name, summary_box)

    @staticmethod
    def add_subdomain_source(ref_component_type, operator_type_name, summary_box):
        physics_type = ref_component_type.collection_type.get_physics_type()
        operator_name_to_type = physics_type.get_operator_name_to_type()
        subdomain_ids = ref_component_type.get_all_geom_ids(is_subdomain=True)
        for subdomain_id in subdomain_ids:
            if subdomain_id.dimension <= 1:
                continue

            if subdomain_id.dimension == 2:
                operator_type_name = 'shell_rotary_acceleration'
            operator_type = operator_name_to_type[operator_type_name]
            new_source_type = ref_component_type.add_new_source(operator_type)
            operator_id = oi.SourceOperatorId(ref_component_type, new_source_type.core.name)
            operator_id.set_mesh_ids([subdomain_id.mesh_id])
            summary_box.add_info(
                'Add ' + operator_type_name + ' source to subdomain ' + str([
                    subdomain_id.mesh_id]))

    @staticmethod
    def assign_edge_port_to_edge_block(ref_component_type, use_non_conforming, relative_tol,
                                       allow_scaling, summary_box):
        ref_mesh = ref_component_type.get_default_mesh()
        for geom_id in ref_mesh.labeled_edges:
            ref_type_geom_id = id_objects.RefTypeGeomId(ref_component_type, geom_id.mesh_id, 1, True)
            success, message = RefComponentTypeGeomUI.auto_assign_port(
                ref_type_geom_id, use_non_conforming=use_non_conforming, relative_tol=relative_tol,
                allow_scaling=allow_scaling)
            if success:
                summary_box.add_info(message)
            else:
                summary_box.add_error('No valid port type found for edge block ' + str(
                    [geom_id.mesh_id]))

    @staticmethod
    def auto_assign_nodeset_using_id_convention(ref_component_type, summary_box, id_parameters):
        node_port_ids_range = id_parameters.node_port_ids_range
        mesh = ref_component_type.get_default_mesh()
        for nodeset_id, node_idxs in mesh.labeled_vertices.items():
            if node_port_ids_range["min"] <= nodeset_id.mesh_id <= node_port_ids_range["max"]:
                RefComponentTypeGeomUI.assign_nodeset_as_node_port(
                    ref_component_type, nodeset_id, node_idxs, summary_box)
            ref_component_type.clear_method_cache()


    @staticmethod
    def create_new_face_port_type(ref_type_geom_id, mesh, collection_type, geom_id,
                                  ref_component_type,
                                  updater, use_non_conforming, relative_tol, allow_scaling,
                                  user_input_port_mesh_ids, summary_box, add_free_nodeset_to_mesh,
                                  extra_stiffener_point=None):
        component_port_mesh_ids = ref_component_type.get_port_mesh_ids()
        exo_data = mesh.create_exo_data_from_geom_ids(
            tuple([geom_id]), geom_id.dimension, component_port_mesh_ids, user_input_port_mesh_ids,
            add_free_nodeset_to_mesh)
        basname = os.path.split(ref_component_type.core.name)[-1]
        new_port_name = basname + "_ss" + str(geom_id.mesh_id)

        temp_exo_file = tempfile.NamedTemporaryFile(delete=False, suffix=".exo")
        temp_exo_file.close()
        exo_data.write(temp_exo_file.name)

        is_shell = ref_component_type.is_shell()
        is_mixed, _, _, solid_block_ids = mesh.check_mixed_elememts()
        if is_mixed:
            # For now, if modeling type is "mixed" and has no solid blocks then assume that
            # port should be shell.
            is_shell |= len(solid_block_ids) == 0
        port_type = port_type_ui.PortTypeUI.create_new_port_type(
            temp_exo_file.name, collection_type, updater, is_shell=is_shell,
            extract_missing_port=True, new_port_name=new_port_name,
            extra_stiffener_point=extra_stiffener_point)

        if updater is not None:
            updater.mesh_like_type_added(port_type)

        success, error_message = ui_objects.RefTypePortUI.apply_port_to_ref_type_geom_id(
            ref_type_geom_id, port_type, use_non_conforming, relative_tol,
            allow_scaling, user_input_port_mesh_ids=user_input_port_mesh_ids,
            add_free_nodeset_to_mesh=add_free_nodeset_to_mesh)
        if success:
            if summary_box is not None:
                summary_box.add_info("-> Created new port type and assigned to sideset " + str(
                    [geom_id.mesh_id]))
        else:
            if summary_box is not None:
                summary_box.add_error("Could not assign new port type to sideset " + str(
                    [geom_id.mesh_id]))
            print(error_message)
        utils.delete_temp_file(temp_exo_file.name)

    @staticmethod
    def create_new_edge_port_type(mesh, geom_id, face_port_type, updater, use_non_conforming,
                                  relative_tol, allow_scaling):
        # Export and edge port type from an edge block of a face port type
        collection_type = face_port_type.get_collection_type()
        exo_data = mesh.create_exo_data_from_geom_ids(tuple([geom_id]), geom_id.dimension)
        basname = os.path.split(face_port_type.core.name)[-1]
        new_port_name = basname + "_block" + str(geom_id.mesh_id)

        temp_exo_file = tempfile.NamedTemporaryFile(delete=False, suffix=".exo")
        temp_exo_file.close()
        exo_data.write(temp_exo_file.name)

        edge_port_type = port_type_ui.PortTypeUI.create_new_port_type(
            temp_exo_file.name, collection_type, updater, is_shell=face_port_type.is_shell(),
            new_port_name=new_port_name)

        port_type_geom_id = id_objects.PortTypeGeomId(
            face_port_type, geom_id.mesh_id, geom_id.dimension)
        success, error_message = ui_objects.RefTypePortUI.apply_edge_port_to_port_type_geom_id(
            port_type_geom_id, edge_port_type, use_non_conforming=use_non_conforming,
            relative_tol=relative_tol, allow_scaling=allow_scaling)
        if not success:
            print(error_message)
        utils.delete_temp_file(temp_exo_file.name)

    @staticmethod
    def create_main_operator_for_1d_block(ref_component_type, block_attributes, id_parameters,
                                          summary_box):
        # For now, Editor only support working on solid and shell subdomain
        # (1D subdomain is treated in other place, 0D is not supported yet)
        ref_mesh = ref_component_type.get_default_mesh()
        block_id_to_names = ref_mesh.mesh_data.block_id_to_names
        collection_type = ref_component_type.get_collection_type()
        subdomain_ids_for_stiffener_selfweight = []
        for beam_block_geom_id in ref_mesh.get_1d_subdomain_geom_ids():
            beam_block_id = beam_block_geom_id.mesh_id
            mesh_subdomain_name = block_id_to_names.get(beam_block_id, str(beam_block_id))
            json_subdomain_name = ref_component_type.get_subdomain_name(beam_block_geom_id)

            if id_parameters.stiffener_ids_range["min"] <= beam_block_id <= \
                    id_parameters.stiffener_ids_range["max"]:
                block_attribute = block_attributes.get(beam_block_id, None)
                is_stiff = RefComponentTypeGeomUI.add_stiffener_type_to_1d_block(
                    ref_component_type, beam_block_id, mesh_subdomain_name,
                    json_subdomain_name, block_attribute, collection_type,
                    summary_box)
                if is_stiff:
                    subdomain_ids_for_stiffener_selfweight.append(beam_block_id)

            elif id_parameters.rigid_beam_ids_range["min"] <= beam_block_id <= \
                    id_parameters.rigid_beam_ids_range["max"]:
                RefComponentTypeGeomUI.add_elasticity_rigid_operator_to_1d_subdomain(
                    ref_component_type, beam_block_id, json_subdomain_name, summary_box)

            elif id_parameters.elasticity_default_beam_ids_range["min"] <= beam_block_id <= \
                    id_parameters.elasticity_default_beam_ids_range["max"]:
                RefComponentTypeGeomUI.add_elasticity_beam_default_operator_to_1d_subdomain(
                    ref_component_type, beam_block_id, collection_type, json_subdomain_name,
                    summary_box)

            else:
                continue

        # Add self-weight source to stiffener
        if len(subdomain_ids_for_stiffener_selfweight) > 0:
            physics_type = collection_type.get_physics_type()
            operator_name_to_type = physics_type.get_operator_name_to_type()
            operator_type = operator_name_to_type[ot.SELF_WEIGHT]
            sw_source_type = ref_component_type.add_new_source(
                operator_type, source_gui_name="Self_weight_for_stiffeners")
            operator_id = oi.SourceOperatorId(ref_component_type, sw_source_type.core.name)
            operator_id.set_mesh_ids(subdomain_ids_for_stiffener_selfweight)
            summary_box.add_info("Add self-weight source to stiffener blocks")

    @staticmethod
    def add_stiffener_type_to_1d_block(ref_component_type, beam_block_id, mesh_subdomain_name,
                                       json_subdomain_name, block_attribute, collection_type,
                                       summary_box):
        operator_name = 'elasticity_beam_stiffener'
        physics_type = collection_type.get_physics_type()
        operator_name_to_type = physics_type.get_operator_name_to_type()

        # mesh_subdomain_name = 'LGA_TUB'
        cross_section_type = "cross_sections/" + mesh_subdomain_name
        if not collection_type.has_component_type(cross_section_type):
            if not summary_box == None:
                summary_box.add_info(
                    'Cannot set 1D subdomain ID [' + str(beam_block_id) + '] as stiffener.')
                return False
                raise user_error_message.UserErrorMessage(
                    "Cross-section type {} does not exist. Please create it first.".format(cross_section_type))
        cross_section_mesh = cross_section_type + "/" + mesh_subdomain_name


        if block_attribute is not None and isinstance(
                block_attribute.material, am.block_attribute_tools.ElasticityMaterial):
            material = block_attribute.material
            young_modulus = material.young_modulus
            poisson_ratio = material.poisson_ratio
            #mass_density = material.mass_density
            # Supposing Young's modulus from (Abaqus) mesh has units Pa, need to convert it to GPa
            # (default gui units system). Need to treat this in a more general way.
            young_modulus /= 1e9
        else:
            print("Mesh does not contain material information for block ID {} "
                  "or material type is not supported yet.".format(beam_block_id))
            young_modulus = 200.
            poisson_ratio = 0.3
            #mass_density = 7850.

        operator_type = operator_name_to_type[operator_name]
        scalars_values = {}
        for scalar in operator_type.parameters:
            scalar_param_name = scalar.name+"_"+str(beam_block_id)
            scalars_values[scalar.name] = scalar_param_name

            if scalar.name == "young_modulus":
                values = [young_modulus]
            elif scalar.name == "poisson_ratio":
                values = [poisson_ratio]
            elif scalar.name == "stiffener_angle_degrees":
                values = [0, 90, 180, -90]
            elif scalar.name == "stiffener_flip_orientation":
                values = [0, 1]
            elif scalar.name == "stiffener_offset_along_shell_normal":
                values = [0.]
            elif scalar.name == "stiffener_offset_orthogonal_to_shell_normal":
                values = [0.]
            else:
                assert False
            non_dim_values = dv.decode_param(
                collection_type, "", np.array(values), type=scalar.units_name)

            discrete_values = []
            for value_idx, non_dim_value in enumerate(non_dim_values):
                discrete_values.append(
                    tp.DiscreteParameterValue(name=str(values[value_idx]), value=non_dim_value))

            scalar_parameter = tp.ParameterType(
                name=scalar_param_name, type=scalar.units_name, discrete_values=discrete_values)
            ref_component_type.core.parameters.append(scalar_parameter)

        ref_component_type.add_stiffener_type_to_1d_block(
            operator_name, beam_block_id, cross_section_mesh, scalars_values,
            subdomain_name=json_subdomain_name, param_type="parameter")
        if summary_box is not None:
            summary_box.add_info(
                'Set 1D subdomain ID [' + str(beam_block_id) + '] as stiffener: ' + str(mesh_subdomain_name))
        return True

    @staticmethod
    def add_elasticity_rigid_operator_to_1d_subdomain(ref_component_type, subdomain_id,
                                                      subdomain_name, summary_box):
        operator_name = 'elasticity_rigid_default'
        operator_definition = tp.OperatorDefn(
            operator_type=operator_name, scalars={}, subdomains=[subdomain_name])
        ref_component_type.core.main_operator_defn.append(operator_definition)
        ref_component_type.core.subdomains.append(tp.SubdomainType(id=subdomain_id, name=subdomain_name))
        summary_box.add_info(
            'Set 1D subdomain ID [' + str(subdomain_id) + '] as rigid beam.')

    @staticmethod
    def add_elasticity_beam_default_operator_to_1d_subdomain(ref_component_type, beam_block_id,
                                                             collection_type, subdomain_name,
                                                             summary_box):
        operator_type_name = "elasticity_beam_default"
        area_value = 1.0e6 # mm^2
        non_dim_area = dv.decode_param(
            collection_type, None, area_value, "mesh_area")
        young_modulus_value = 200.0 # GPa
        non_dim_young_modulus = dv.decode_param(
            collection_type, None, young_modulus_value, "young_modulus")
        moment_of_inertia_1_value = 0.0833333e12 # mm^4
        non_dim_moment_of_inertia_1 = dv.decode_param(
            collection_type, None, moment_of_inertia_1_value, "moment_of_inertia_1")
        moment_of_inertia_2_value = 0.0833333e12 # mm^4
        non_dim_moment_of_inertia_2 = dv.decode_param(
            collection_type, None, moment_of_inertia_2_value, "moment_of_inertia_2")
        torsion_constant_value = 0.14237e12 # mm^4
        non_dim_torsion_constant = dv.decode_param(
            collection_type, None, torsion_constant_value, "torsion_constant")
        offset_value = 0.0
        non_dim_offset = dv.decode_param(
            collection_type, None, offset_value, "mesh_length")
        poisson_ratio_value = non_dim_poisson_ratio = 0.3

        scalar_values = {
            "area": ("parameter", "area_" + str(beam_block_id)),
            "offset": ("parameter", "offset_" + str(beam_block_id)),
            "moment_of_inertia_1": ("parameter", "moment_of_inertia_1_" + str(beam_block_id)),
            "moment_of_inertia_2": ("parameter", "moment_of_inertia_2_" + str(beam_block_id)),
            "young_modulus": ("parameter", "young_modulus_" + str(beam_block_id)),
            "torsion_constant": ("parameter", "torsion_constant_" + str(beam_block_id)),
            "poisson_ratio": ("parameter", "poisson_ratio_" + str(beam_block_id))}

        scalars = {}
        for name, scalar in scalar_values.items():
            scalar_type, scalar_value = scalar
            scalars[name] = (tp.ScalarValue(type=scalar_type, value=str(scalar_value)))

        operator_definition = tp.OperatorDefn(
            operator_type=operator_type_name, scalars=scalars, subdomains=[])
        operator_definition.subdomains.append(subdomain_name)
        ref_component_type.core.main_operator_defn.append(operator_definition)
        if summary_box is not None:
            summary_box.add_info(
                'Add main operator ' + operator_type_name +
                ' to subdomain [' + str(beam_block_id) + ']')
        # Add scalar parameter to ref_component_type.core.parameters
        for param_type in scalar_values:
            param = tp.ParameterType(
                name=param_type+"_"+str(beam_block_id), type=param_type,
                discrete_values=[tp.DiscreteParameterValue(
                    name=str(locals()[param_type+"_value"]), value=locals()[
                        "non_dim_"+param_type])])
            ref_component_type.core.parameters.append(param)

    @staticmethod
    def update_point_subdomain_ids(ref_component_type):
        # Update point_subdomain_ids
        # Two kinds of node will be assigned as PointSubdomainIdType:
        #  (1) nodeset
        #  (2) node that belong to implicit wire-basket edge
        point_subdomain_ids = []
        ref_mesh = ref_component_type.get_default_mesh()
        port_mesh_ids = ref_component_type.get_port_mesh_ids()
        candidate_point_subdomain_ids = ref_mesh.get_point_subdomain_ids(port_mesh_ids)
        for mesh_node_idx in candidate_point_subdomain_ids:
            vertex_coords = ref_mesh.mesh_data.coords[mesh_node_idx]
            vertex_subdomain_id = int(ref_mesh.mesh_coord_subdomains[mesh_node_idx])
            point_subdomain_id = tp.PointSubdomainIdType(
                point=[float(x) for x in vertex_coords], subdomain_id=vertex_subdomain_id)
            point_subdomain_ids.append(point_subdomain_id)

        if len(point_subdomain_ids) > 0:
            ref_component_type.core.point_subdomain_ids = point_subdomain_ids

    @staticmethod
    def check_edge_and_stiffener_intersection(mesh, edge_port_id, stiffener_infos):
        mesh_node_normals = mesh.compute_mesh_node_normals(mesh.draw_coords)
        _, _, _, edge_bounding_boxes, _, _, _, _, edge_centroids, _ = mesh.compute_bbox_info(mesh.draw_coords)
        for stiffener_info in stiffener_infos:
            stiffener_geom_id = stiffener_info.get_geom_ids()[0]
            intersecting_nodes = mesh.check_if_two_edge_geom_ids_intersect(
                edge_port_id, stiffener_geom_id)
            # We expect that a stiffener cut an edge port at one node at maximum
            assert len(intersecting_nodes) <= 1
            if len(intersecting_nodes) > 0:
                middle_point = edge_centroids[edge_port_id]
                normal_vector = mesh_node_normals[intersecting_nodes[0]]
                edge_bb_radius = edge_bounding_boxes[edge_port_id].radius()
                extra_node = middle_point + edge_bb_radius*normal_vector
                return True, extra_node
        return False, None

    @staticmethod
    def check_edge_and_stiffener_intersection_2(mesh, edge_port_id):
        try:
            mesh_node_normals = mesh.compute_mesh_node_normals(mesh.draw_coords)
            _, _, _, edge_bounding_boxes, _, _, _, _, edge_centroids, _ = mesh.compute_bbox_info(mesh.draw_coords)
            edge_nodes = mesh.get_sideset_node_idx(edge_port_id)
            middle_point = edge_centroids[edge_port_id]
            normal_vector = mesh_node_normals[edge_nodes[0]]
            edge_bb_radius = edge_bounding_boxes[edge_port_id].radius()
            extra_node = middle_point + edge_bb_radius * normal_vector
            return True, extra_node
        except:
            return False, None

    @staticmethod
    def calculate_n_dofs(ref_component_type, physics_name, summary_box):
        ref_mesh = ref_component_type.get_default_mesh()
        modeling_data = ref_component_type.core.modeling_data
        n_dofs = new_type.calculate_n_dofs(ref_mesh, physics_name, modeling_data)
        if n_dofs > 500000:
            summary_box.add_info("Number of DOFs: " + str(n_dofs))
            summary_box.add_info(
                "Warning: component is larger than 500,000 DOFs so Akselos Dashboard "
                "may not be able to train. Please split it into smaller components.",
                color='red')
        else:
            summary_box.add_info("Number of DOFs: " + str(n_dofs))


class GeomEditGraphicsScene(ref_component_type_graphics.RefComponentTypeGraphicsBase):
    def __init__(self, istate, canvas, ref_component_type_geom_id):
        self.ref_component_type_geom_id = ref_component_type_geom_id
        ref_component_type_graphics.RefComponentTypeGraphicsBase.__init__(
            self, istate, canvas, ref_component_type_geom_id.ref_component_type)
        self.highlighted_side_idx = -1
        self.dirty = True
        self.graphics_body_cache = None
        # Eh, we use our own set of selected items.  The default set gets observed by the context
        # box, so changing the selection causes the context box to update, which is very slow.
        self.highlight = set()
        self.selection = set()

    @staticmethod
    @utils.memoized(weak_key=True, weak_key_arg=2)
    def create(istate, canvas, ref_component_type, *other_args):
        # We want the memoize to be weak key on the ref component type.  So we split the
        # ref_component_type_geom_id into its tuple when calling this function.
        ref_component_type_geom_id = id_objects.RefTypeGeomId.from_tuple(
            ref_component_type, *other_args)
        return GeomEditGraphicsScene(istate, canvas, ref_component_type_geom_id)

    def do_updateGL(self, gl_manager):
        self.graphics_body_cache = GeomEditComponentGraphicsBody.create(
          self.default_mesh, self.ref_component_type_geom_id)

        self.dirty = False
        self.default_mesh = self.create_default_mesh()
        self.bodies.clear()
        self.selection.clear()
        self.highlight.clear()

        for geom_id in self.ref_component_type_geom_id.get_geom_ids():
            geom_ref = fr.GeomRef(self.ref_component_type, geom_id)
            self.selection.add(geom_ref)

        self.bodies[self.ref_component_type] = self.graphics_body_cache
        if self.highlighted_side_idx != -1:
            self.graphics_body_cache.update_highlight_ids(self.highlighted_side_idx)

        for highlight_geom_id in self.graphics_body_cache.highlight_geom_ids:
            self.highlight.add(fr.GeomRef(self.ref_component_type, highlight_geom_id))

        # Added wirebasket graphic body
        is_wirebasket = self.ref_component_type_geom_id.check_wirebasket()[0]
        if is_wirebasket:
            if geom_id.dimension == 0:
                vertices, connection = \
                    self.ref_component_type_geom_id.get_wirebasket_graphic_data()
                self.bodies[geom_id] = gb.WireBasketItemGraphicsBody.create(vertices, connection)
            elif geom_id.dimension == 1:
                vertices, connection = self.ref_component_type_geom_id.\
                    get_wirebasket_graphic_data()
                self.bodies[geom_id] = gb.WireBasketItemGraphicsBody.create(vertices, connection)

        ref_component_type_graphics.RefComponentTypeGraphicsBase.updateGL(self, gl_manager)


class GeomEditComponentGraphicsBodyData(ga.GraphicsBodyData):
    def __init__(self, mesh):
        ga.GraphicsBodyData.__init__(self)
        self.mesh = mesh
        self.mapped_mesh_graphics_body_data = gb.MappedMeshGraphicsBodyData.create(mesh)
        self.vertices = self.mapped_mesh_graphics_body_data.vertices
        self.normals = self.mapped_mesh_graphics_body_data.normals
        self.extra_graphics_face_datas = {}

    def get_element_index_data(self, element_indices_key):
        # We return the regular face datas for the mesh, and we add some extra ones for
        # highlighted mesh elements.
        mapped_mesh_element_index_data = self.mapped_mesh_graphics_body_data.get_element_index_data(
            element_indices_key)
        element_indices = mapped_mesh_element_index_data.element_indices
        graphics_face_datas = dict(mapped_mesh_element_index_data.graphics_face_datas)
        graphics_face_datas.update(self.extra_graphics_face_datas)
        element_index_data = ga.ElementIndexData(element_indices, graphics_face_datas)
        return element_index_data


class GeomEditComponentGraphicsBody(ga.GraphicsBody):
    def __init__(self, mesh, ref_type_geom_id):
        ga.GraphicsBody.__init__(self)
        mesh_dimension = ref_type_geom_id.get_dimension()
        if mesh_dimension == 2:
            selection_mode = sm.SelectionModes.MeshElements
        elif mesh_dimension == 1:
            selection_mode = sm.SelectionModes.MeshElements
        else:
            selection_mode = sm.SelectionModes.MeshElements

        self.mesh = mesh
        self.graphics_body_data = GeomEditComponentGraphicsBodyData(mesh)
        self.unsorted_triangle_idxs = self.get_unsorted_triangle_idxs(mesh)
        geom_ids = set(gb.get_geom_ids_for_mesh(mesh, selection_mode))
        if mesh_dimension == 1:
            labeled_edge_ids = set(ref_type_geom_id.get_geom_ids())
            labeled_node_ids = None
        elif mesh_dimension == 0:
            labeled_node_ids = set(ref_type_geom_id.get_geom_ids())
            labeled_edge_ids = None
        else:
            labeled_edge_ids = None
            labeled_node_ids = None

        edge_id_to_color = None
        if labeled_edge_ids is not None:
            edge_id_to_color = {edge_id: None for edge_id in labeled_edge_ids}

        pick_object = ref_type_geom_id.ref_component_type
        self.graphics_face_instances = gb.get_graphics_face_instances(
            geom_ids, pick_object, selection_mode=selection_mode,
            edge_id_to_color=edge_id_to_color, labeled_node_ids=labeled_node_ids)
        self.pick_object = pick_object
        self.back = False
        self.highlight_geom_ids = []

    @staticmethod
    def get_unsorted_triangle_idxs(mesh):
        # The triangle_indices are a list of triangles grouped so that they are contiguous for a
        # single rendered item.  The unsorted_triangle_idxs member maps from these sorted triangles
        # to the index of the triangle according to the mesh's 'triangles' field.
        triangle_idxs_map = mesh.get_surface_id_to_triangle_idxs()

        triangle_idxs_list = []
        for surface_id, triangle_idxs in triangle_idxs_map.items():
            triangle_idxs_list.append(triangle_idxs)

        unsorted_triangle_idxs = utils.concat_default(triangle_idxs_list)
        return unsorted_triangle_idxs

    def update_highlight_ids(self, side_idx):
        # Here is some rather hacky stuff to highlight a single side without having to create
        # separate faces for all the sides in the mesh (which is slow).  For a given side, we
        # find which triangles belong to that side.  We then find the index of each of those
        # triangles in the list of triangles to render, and add a GraphicsFace for
        # each triangle.
        triangle_flags = ms.mesh_tools.get_triangle_flags_from_side_idxs(
            np.array([side_idx]), self.mesh.triangle_side_idxs)
        triangle_idxs = np.where(triangle_flags)[0]
        start_stops = []
        for triangle_idx in triangle_idxs:
            match_idxs = np.where(self.unsorted_triangle_idxs - triangle_idx == 0)[0]
            if len(match_idxs) == 1:
                start_stops.append((match_idxs[0]*3, match_idxs[0]*3+3))
        self.highlight_geom_ids = []

        self.graphics_body_data.extra_graphics_face_datas.clear()
        for i, (start_idx, stop_idx) in enumerate(start_stops):
            highlight_id = -i-1
            highlight_geom_id = ms.geom_id.GeomId(highlight_id, 2, False)
            self.graphics_body_data.extra_graphics_face_datas[highlight_geom_id] = \
                ga.GraphicsFaceData(int(start_idx), int(stop_idx))
            self.highlight_geom_ids.append(highlight_geom_id)

    @staticmethod
    def create(mesh, ref_type_geom_id):
        if ref_type_geom_id.is_named_geom():
            surface_ids = ref_type_geom_id.get_geom_ids()
            colors = {}
            for surface_id in surface_ids:
                colors[surface_id] = (0.8, 0.5, 0.5, 1.0)
        else:
            pass

        return GeomEditComponentGraphicsBody(mesh, ref_type_geom_id)


class RefComponentTypeNodeHandler(hn.ActionObjectBase):
    def __init__(self, istate, canvas, ref_component_type_geom_id, node_position=None,
                 node_index=None, restore_view=True):
        hn.ActionObjectBase.__init__(self, istate)
        self.canvas = canvas
        self.ref_component_type_geom_id = ref_component_type_geom_id
        bounding_box = self.get_graphics_scene().get_bounding_box()
        if restore_view:
            istate.view.view_best_for_bbox(bounding_box)
        self.selector = selector.ComponentSelector.create_for_ref_component_type(
            self.istate, self.ref_component_type_geom_id.ref_component_type)
        self._update_is_editing()
        self.geom_id = self.ref_component_type_geom_id.get_single_geom_id()
        self.node_position = node_position
        self.node_index = node_index

    def _update_is_editing(self):
        ui = rf.RefComponentTypeUI.get(self.ref_component_type_geom_id.ref_component_type)
        self.is_editing = \
            ui.get_is_editing() and self.ref_component_type_geom_id.get_port_idx() is None

    def get_subject(self):
        return self.ref_component_type_geom_id

    def get_graphics_scene(self):
        return GeomEditGraphicsScene.create(
            self.istate, self.canvas, *self.ref_component_type_geom_id.as_tuple())

    def mouseMoveEvent(self, x, y):
        if not self.is_editing:
            return self

        self.node_position = None
        self.node_index = None

        key, model_point, model_normal, triangle_idx, _ = self.selector.get_mouse_hit(x, y)
        if key is None:
            return self

        if triangle_idx is None:
            return self

        mesh = self.ref_component_type_geom_id.ref_component_type.get_default_mesh()
        selected_face = mesh.face[triangle_idx]
        d_0 = (pt.Point(mesh.node[selected_face[0]]) - model_point).mag()
        d_1 = (pt.Point(mesh.node[selected_face[1]]) - model_point).mag()
        d_2 = (pt.Point(mesh.node[selected_face[2]]) - model_point).mag()
        if d_0 < d_1 and  d_0 < d_2:
            selected_node = 0
        elif d_1 < d_2:
            selected_node = 1
        else:
            selected_node = 2
        node_index = selected_face[selected_node]
        if node_index < len(mesh.node):
            vertex = mesh.node[node_index]
            self.node_position = pt.Point(vertex)
            self.node_index = mesh.draw_coord_info['mesh_node_idx'][node_index]

        return RefComponentTypeNodeHandler(self.istate,
            self.canvas, self.ref_component_type_geom_id, node_position=self.node_position,
            node_index=self.node_index, restore_view=False)

    def widgets(self, graphics_scene):
        if self.node_position is None:
            return []

        inspection_sphere = ws.SphereWidget(
            self.node_position, 0.15, (0.86, 0.73, 0.4, 1.0), auto_scale=True)
        inspection_sphere.should_clip = False
        return [inspection_sphere]

    def leftPressEvent(self):
        if not self.is_editing:
            return self

        if isinstance(self.node_index, np.int32):
            mesh = self.ref_component_type_geom_id.ref_component_type.get_default_mesh()
            mesh.set_nodeset(self.geom_id, self.node_index)

            if mesh.mapped_mesh in gb.MappedMeshGraphicsBodyData.instance_cache:
                del gb.MappedMeshGraphicsBodyData.instance_cache[mesh.mapped_mesh]
            graphics_scene = self.get_graphics_scene()
            graphics_scene.body_graphics_changed()

            ref_component_type_ui = rf.RefComponentTypeUI.get(
                self.ref_component_type_geom_id.ref_component_type)
            updater = self.canvas.istate.main.updater
            ref_component_type_ui.apply_changes(updater, mesh_modified=True)
            self._update_is_editing()

        return RefComponentTypeNodeHandler(self.istate,
            self.canvas, self.ref_component_type_geom_id, node_position=self.node_position,
            node_index=self.node_index, restore_view=False)

    def wheelEvent(self, delta):
        self.zoom(delta)
        return self

    def cursor(self):
        if self.is_editing:
            return qt.Qt.CrossCursor
        return hn.eye_cursor()


class RefComponentTypeEdgeHandler(hn.ActionObjectBase):
    def __init__(self, istate, canvas, ref_component_type_geom_id):
        hn.ActionObjectBase.__init__(self, istate)
        self.canvas = canvas
        self.ref_component_type_geom_id = ref_component_type_geom_id
        bounding_box = self.get_graphics_scene().get_bounding_box()
        istate.view.view_best_for_bbox(bounding_box)

    def get_subject(self):
        return self.ref_component_type_geom_id

    def get_graphics_scene(self):
        return GeomEditGraphicsScene.create(
            self.istate, self.canvas, *self.ref_component_type_geom_id.as_tuple())

    def mouseMoveEvent(self, x, y):
        return self

    def leftPressEvent(self):
        return self

    def wheelEvent(self, delta):
        self.zoom(delta)
        return self.view_changed()

    def cursor(self):
        return hn.eye_cursor()


class RefComponentTypeFaceHandler(hn.ActionObjectBase):
    def __init__(self, istate, canvas, ref_component_type_surface_id, is_adding=None,
                 last_triangle_idx=-1, adjust_bounding_box=True):
        hn.ActionObjectBase.__init__(self, istate)
        self.canvas = canvas
        self.ref_component_type_surface_id = ref_component_type_surface_id
        self._update_is_editing()
        self.is_adding = is_adding
        self.selector = selector.ComponentSelector.create_for_ref_component_type(
            self.istate, self.ref_component_type_surface_id.ref_component_type)
        self.last_triangle_idx = last_triangle_idx

        if adjust_bounding_box:
            bounding_box = self.get_graphics_scene().get_bounding_box()
            self.istate.view.view_best_for_bbox(bounding_box)

    def _update_is_editing(self):
        ui = rf.RefComponentTypeUI.get(self.ref_component_type_surface_id.ref_component_type)
        self.is_editing = ui.get_is_editing() and self.ref_component_type_surface_id.get_port_idx() is None

    def view_changed(self):
        # Need to remake the selector cache
        return RefComponentTypeFaceHandler(
            self.istate, self.canvas, self.ref_component_type_surface_id, self.is_adding, adjust_bounding_box=False)

    def cursor(self):
        if self.is_editing:
            return qt.Qt.CrossCursor
        return hn.eye_cursor()

    def get_graphics_scene(self):
        return GeomEditGraphicsScene.create(
            self.istate, self.canvas, *self.ref_component_type_surface_id.as_tuple())

    def widgets(self, graphics_scene):
        widgets = []
        return widgets

    def rightPressEvent(self, event=None):
        return self

    def get_highlighted_triangle_idx(self):
        x, y = self.istate.x(), self.istate.y()
        _, model_point, model_normal, triangle_idx, _ = self.selector.get_mouse_hit(x, y)
        return triangle_idx

    def leftDoubleClickEvent(self):
        if not self.is_editing:
            return self
        ref_component_type = self.ref_component_type_surface_id.ref_component_type

        surface_ids = self.ref_component_type_surface_id.get_geom_ids()
        if len(surface_ids) != 1:
            # We don't currently handling editing a named surface.
            return self.copy_with(self.is_adding)
        surface_id, = surface_ids

        triangle_idx = self.last_triangle_idx
        if triangle_idx is None or triangle_idx == -1:
            return self.copy_with(self.is_adding)

        side_idx = ref_component_type.get_default_mesh().triangle_side_idxs[triangle_idx]
        mesh = ref_component_type.get_default_mesh()
        on_side_idxs = mesh.labeled_surfaces[surface_id]
        on_flags = np.zeros(len(mesh.sides), dtype=np.bool)
        on_flags[on_side_idxs] = True
        if self.is_adding is not False:
            on_flags[side_idx] = False
            on_side_idxs = np.where(on_flags)[0]
            connected_side_idxs = mesh.get_connected_side_idxs(side_idx, on_side_idxs)
            assert side_idx in connected_side_idxs
            new_side_idxs = np.concatenate((on_side_idxs, connected_side_idxs))
        else:
            on_flags[side_idx] = True
            off_side_idxs = np.where(~on_flags)[0]
            connected_side_idxs = mesh.get_connected_side_idxs(side_idx, off_side_idxs)
            assert side_idx in connected_side_idxs
            new_side_idxs = on_side_idxs[~np.in1d(on_side_idxs, connected_side_idxs)]

        mesh.set_sideset(surface_id, new_side_idxs)
        if len(new_side_idxs) > 0:
            # After modifying a surface definition, need to update mesh info
            mesh.set_bbox(mesh.draw_coords)
        ref_component_type_ui = rf.RefComponentTypeUI.get(ref_component_type)
        updater = self.canvas.istate.main.updater
        ref_component_type_ui.apply_changes(updater, mesh_modified=True)
        if mesh.mapped_mesh in gb.MappedMeshGraphicsBodyData.instance_cache:
            del gb.MappedMeshGraphicsBodyData.instance_cache[mesh.mapped_mesh]
        graphics_scene = self.get_graphics_scene()
        graphics_scene.body_graphics_changed()
        return self.copy_with(None)

    def leftReleaseEvent(self):
        return self

    def leftPressEvent(self):
        if not self.is_editing:
            return self
        return self.handle_left_down(is_adding=None)

    def mouseMoveEvent(self, x, y):
        # We call this here to avoid having to send a notification from the context box to the handler
        # when the edition state changes
        self._update_is_editing()
        if not self.is_editing:
            return self
        if self.istate.left():
            next_handler = self.handle_left_down(self.is_adding)
        else:
            # Can a mouse move event occur between a left release event and double click event?
            # Hmm.
            ref_component_type = self.ref_component_type_surface_id.ref_component_type
            triangle_idx = self.get_highlighted_triangle_idx()
            if triangle_idx is None:
                side_idx = -1
            else:
                side_idx = ref_component_type.get_default_mesh().triangle_side_idxs[triangle_idx]
            graphics_scene = self.get_graphics_scene()
            if graphics_scene.highlighted_side_idx != side_idx:
                graphics_scene.highlighted_side_idx = side_idx
                graphics_scene.body_graphics_changed()

            next_handler = self.copy_with(self.is_adding, self.last_triangle_idx)
        return next_handler

    def handle_left_down(self, is_adding):
        triangle_idx = self.get_highlighted_triangle_idx()
        if triangle_idx is None:
            return self.copy_with(is_adding, self.last_triangle_idx)

        ref_component_type = self.ref_component_type_surface_id.ref_component_type
        surface_ids = self.ref_component_type_surface_id.get_geom_ids()
        if len(surface_ids) != 1:
            # We don't currently handling editing a named surface.
            return self.copy_with(is_adding, self.last_triangle_idx)
        surface_id, = surface_ids

        updater = self.canvas.istate.main.updater
        new_is_adding = self.hit_triangle_idx_on_surface(
            ref_component_type, surface_id, triangle_idx, is_adding, updater)

        graphics_scene = self.get_graphics_scene()
        graphics_scene.body_graphics_changed()
        return self.copy_with(new_is_adding, triangle_idx)

    def copy_with(self, is_adding=None, last_triangle_idx=-1):
        return RefComponentTypeFaceHandler(
            self.istate, self.canvas, self.ref_component_type_surface_id, is_adding,
            last_triangle_idx, adjust_bounding_box=False)

    @staticmethod
    def hit_triangle_idx_on_surface(ref_component_type, surface_id, triangle_idx,
                                    is_adding, updater):
        side_idx = ref_component_type.get_default_mesh().triangle_side_idxs[triangle_idx]
        mesh = ref_component_type.get_default_mesh()
        current_side_idxs = mesh.labeled_surfaces[surface_id]
        has_mask = current_side_idxs == side_idx
        already_has = np.any(has_mask)
        # If is_adding is not None (i.e. it's False or True), then we already know whether
        # we are adding or removing elements and we should continue doing that.  Otherwise,
        # set whether we are adding or removing elements based on whether the current side is
        # contained in the sideset.
        if is_adding is False:
            new_side_idxs = current_side_idxs[~has_mask]
        elif is_adding is True:
            new_side_idxs = np.concatenate((current_side_idxs, [side_idx]))
        elif already_has:
            is_adding = False
            new_side_idxs = current_side_idxs[~has_mask]
        else:
            is_adding = True
            new_side_idxs = np.concatenate((current_side_idxs, [side_idx]))

        mesh.set_sideset(surface_id, new_side_idxs)
        if len(new_side_idxs) > 0:
            # After modifying a surface definition, need to update mesh info
            mesh.set_bbox(mesh.draw_coords)
        ref_component_type_ui = rf.RefComponentTypeUI.get(ref_component_type)
        ref_component_type_ui.apply_changes(updater, mesh_modified=True)
        if mesh.mapped_mesh in gb.MappedMeshGraphicsBodyData.instance_cache:
            del gb.MappedMeshGraphicsBodyData.instance_cache[mesh.mapped_mesh]
        return is_adding

    def wheelEvent(self, delta):
        self.zoom(delta)
        return self.view_changed()

    def get_subject(self):
        return self.ref_component_type_surface_id

    def status_message(self):
        return "Click to add/remove elements, drag to add/remove multiple elements, " \
               "double click to add/remove region."


def _concat_default(a, dtype=np.int32):
    if len(a) == 0:
        return np.array([], dtype=dtype)
    else:
        return np.concatenate(a)
