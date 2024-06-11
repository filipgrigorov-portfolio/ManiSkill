import os.path as osp
from pathlib import Path
from typing import List, Optional
import json

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_UNIQUE_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


def _customize_table(
    table_model_path: Path, color: Optional[List[float]] = None, remove_texture=False
) -> Path:
    """
    Create a new .glb file for table with new color.
    Return the path to the customized table.
    Args:
        color: RGBA, should be in range [0, 1]
    """
    customized_table_model_path = (
        table_model_path.parent
        / f"{table_model_path.stem}_customized{table_model_path.suffix}"
    )
    print(f"Using: customized_table_model_path")
    from pygltflib import GLTF2, Material

    # Load the GLB file
    gltf = GLTF2().load(table_model_path)

    # Assuming want to change the color of the first material (true for table)
    material = gltf.materials[0]

    # Check if the material has a PBR metallic roughness property
    if not material.pbrMetallicRoughness:
        raise ValueError("Need pbrMetallicRoughness property")

    if remove_texture:
        # Remove preexisting wooden-texture
        material.pbrMetallicRoughness.baseColorTexture = None

    # Set the base color to the desired RGBA values
    material.pbrMetallicRoughness.baseColorFactor = color

    # Save the customized GLB file
    gltf.save(customized_table_model_path)
    return customized_table_model_path


def _preview_table(table_model_path: Path, view_image=False) -> Path:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(table_model_path))
    imported_objects = bpy.context.selected_objects

    bpy.ops.object.camera_add(location=(0.74277564, -2.32305479, -1.68291426))
    camera = bpy.context.object
    camera.rotation_euler = (
        0.18427537381649017,
        -2.2037839889526367,
        1.7085068225860596,
    )
    bpy.context.scene.camera = camera  # Set the camera as the active camera

    # This light source shines on the table top, increase or decrease intensity as desired.
    bpy.ops.object.light_add(type="SUN", location=(5, -5, 5))
    light = bpy.context.object
    light.data.energy = 150

    # Additional light from below to improve look
    bpy.ops.object.light_add(type="POINT", location=(0, 0, 5))
    point_light = bpy.context.object
    point_light.data.energy = 1000

    # Ensure there is a world in the scene
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("NewWorld")
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1)  # Set background color to grey

    # Set the render settings
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(
        table_model_path.parent / f"{table_model_path.stem}_preview.png"
    )
    # Render the image
    bpy.ops.render.render(write_still=True)

    print(f"Preview saved to: {bpy.context.scene.render.filepath}")

    if view_image:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(bpy.context.scene.render.filepath)
        imgplot = plt.imshow(img)
        plt.show()


# TODO (stao): make the build and initialize api consistent with other scenes
class TableSceneBuilder(SceneBuilder):
    robot_init_qpos_noise: float = 0.02

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(
            Path(__file__).parent.parent.parent.parent.parent / "table_config.json"
        ) as f: # Project root directory / table_config.json
            self.config = json.load(f)

    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")
        if len(self.config) > 0:
            table_model_file = _customize_table(
                Path(table_model_file),
                color=self.config["color"],
                remove_texture=self.config["remove_texture"],
            )
            _preview_table(table_model_file, view_image=True)
            table_model_file = str(table_model_file)
        print(f"INFO: Using {table_model_file} for the table model")
        scale = 1.75

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder.add_nonconvex_collision_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=table_pose,
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]

        self.ground = build_ground(self.scene, altitude=-self.table_height)
        self.table = table
        self._scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -self.table_height], q=euler2quat(0, 0, np.pi / 2))
        )
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_UNIQUE_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass

    @property
    def scene_objects(self):
        return self._scene_objects

    @property
    def movable_objects(self):
        raise AttributeError(
            "For TableScene, additional movable objects must be added and managed at Task-level"
        )
