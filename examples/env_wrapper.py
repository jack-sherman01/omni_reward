'''
this is Environment wrapper that replaces rewards with OmniReward
connect OmniReward with your task environment
'''
# import gym
import os
import numpy as np
import numpy.typing as npt
from PIL import Image
from omni_reward.reward.interface import OmniRewardInterface


class OmniRewardWrapper:
    """Environment wrapper that replaces rewards with OmniReward."""
    
    def __init__(
        self, 
        env, 
        captioner, 
        text_encoder, 
        goal_text, 
        use_subgoals=True,
        # API key
        openai_api_key: str | None = None,
        # Image saving parameters
        save_images: bool = True,
        image_save_dir: str | None = None,
        task_name: str = "default_task",
        # Camera settings
        camera_name: str = "corner2",
        # VLM call frequency settings
        vlm_call_interval: int = 10,  # Call VLM every N steps
        use_interpolated_reward: bool = True,  # Interpolate reward between VLM calls
    ):
        # Set API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.env = env
        # Allow goal change if using subgoals
        self.reward_interface = OmniRewardInterface(
            captioner, 
            text_encoder,
            allow_goal_change=use_subgoals
        )
        self.goal_text = goal_text
        self.use_subgoals = use_subgoals
        self.task_name = task_name
        self.camera_name = camera_name
        
        # VLM call frequency settings
        self.vlm_call_interval = vlm_call_interval
        self.use_interpolated_reward = use_interpolated_reward
        self._last_vlm_reward = 0.0
        self._last_vlm_result = None
        self._steps_since_vlm_call = 0
        
        # Image saving settings - automatically include task name
        self.save_images = save_images
        self.image_save_dir = image_save_dir if image_save_dir else f"./images/{task_name}"
        if self.save_images:
            os.makedirs(self.image_save_dir, exist_ok=True)
        
        self._timestep = 0
        self._episode = 0
    
    @property
    def action_space(self):
        """Proxy the action_space of the underlying environment"""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Proxy the observation_space of the underlying environment"""
        return self.env.observation_space
    
    def __getattr__(self, name):
        """Proxy all other attributes to the underlying environment"""
        return getattr(self.env, name)
        
    def _get_image(self, camera_name: str | None = None):
        """Get the current image from the environment
        
        Args:
            camera_name: Camera name, options:
                - "corner": Corner view (default)
                - "corner2": Side view
                - "corner3": Another side view
                - "behindGripper": Behind the gripper view
                - "topview": Top-down view
                - "frontview": Front view (custom)
                - None: Use the camera set during initialization
        """
        cam = camera_name or self.camera_name
        
        # MetaWorld uses mujoco, need to call sim.render directly
        if hasattr(self.env, 'mujoco_renderer'):
            # Newer MetaWorld uses mujoco_renderer
            try:
                import mujoco
                
                # Get the model and data from the renderer
                model = self.env.mujoco_renderer.model
                data = self.env.mujoco_renderer.data
                
                # Create rendering context and render
                width = self.env.width if hasattr(self.env, 'width') else 480
                height = self.env.height if hasattr(self.env, 'height') else 480
                
                renderer = mujoco.Renderer(model, height, width)
                
                # If using custom frontview, manually set camera parameters
                if cam == "frontview":
                    renderer.update_scene(data)
                    renderer.scene.camera.lookat[:] = [0.0, 0.6, 0.2]
                    renderer.scene.camera.distance = 1.2
                    renderer.scene.camera.azimuth = 180
                    renderer.scene.camera.elevation = -25
                else:
                    try:
                        renderer.update_scene(data, camera=cam)
                    except Exception as e:
                        print(f"Camera {cam} not found, using default: {e}")
                        renderer.update_scene(data)
                
                img = renderer.render()
                renderer.close()
                
                # mujoco renders images upside down, need to flip
                img = img[::-1]
                return img
            except Exception as e:
                print(f"mujoco_renderer failed for camera {cam}: {e}")
        
        if hasattr(self.env, 'sim'):
            # Older MetaWorld uses sim (mujoco_py)
            try:
                width = self.env.width if hasattr(self.env, 'width') else 480
                height = self.env.height if hasattr(self.env, 'height') else 480
                
                if cam == "frontview":
                    print("Warning: frontview not supported with mujoco_py, using corner")
                    cam = "corner"
                
                img = self.env.sim.render(
                    width=width,
                    height=height,
                    camera_name=cam,
                    mode='offscreen'
                )
                # mujoco_py also renders images upside down
                return img[::-1]
            except Exception as e:
                print(f"sim.render failed for camera {cam}: {e}")
        
        # Finally, try calling env.render()
        print(f"Warning: Using default render(), camera '{cam}' may not be applied")
        return self.env.render()
    
    def _get_multi_view_images(self) -> dict:
        """Get images from multiple camera views"""
        cameras = ["corner", "corner2", "corner3", "topview", "behindGripper"]
        images = {}
        for cam in cameras:
            try:
                images[cam] = self._get_image(camera_name=cam)
            except Exception as e:
                print(f"Failed to render camera {cam}: {e}")
        return images
    
    def _save_image(self, image: npt.NDArray[np.uint8], step: int) -> None:
        """Save image to local directory"""
        filename = f"episode_{self._episode:04d}_step_{step:04d}.png"
        filepath = os.path.join(self.image_save_dir, filename)
        img = Image.fromarray(image)
        img.save(filepath)
    
    def _should_call_vlm(self) -> bool:
        """Determine if VLM should be called this step"""
        return self._steps_since_vlm_call >= self.vlm_call_interval
    
    def _compute_interpolated_reward(self) -> float:
        """Compute interpolated reward between VLM calls
        
        Uses the last VLM reward, optionally with decay or other strategies.
        """
        if self.use_interpolated_reward:
            # Strategy 1: Use last reward (simple)
            return self._last_vlm_reward
            
            # Strategy 2: Decay reward over time (alternative)
            # decay_factor = 0.95 ** self._steps_since_vlm_call
            # return self._last_vlm_reward * decay_factor
            
            # Strategy 3: Use zero between calls (sparse)
            # return 0.0
        else:
            return 0.0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._timestep = 0
        self._episode += 1
        
        # Reset VLM call tracking
        self._steps_since_vlm_call = self.vlm_call_interval  # Force VLM call on first step
        self._last_vlm_reward = 0.0
        self._last_vlm_result = None
        
        # Reset the reward interface state
        self.reward_interface.reset_episode()
        
        # Get the initial image (using the specified camera)
        initial_image = self._get_image(camera_name=self.camera_name)
        
        # Save the initial image
        if self.save_images and initial_image is not None:
            self._save_image(initial_image, step=0)
        
        if self.use_subgoals:
            self.reward_interface.start_episode_with_subgoals(
                self.goal_text, 
                initial_image=initial_image,
                auto_decompose=True
            )
        else:
            self.reward_interface.start_episode(self.goal_text, initial_image=initial_image)
        
        return obs, info
    
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self._timestep += 1
        self._steps_since_vlm_call += 1
        
        # Get the current image (using the specified camera)
        current_image = self._get_image(camera_name=self.camera_name)
        
        # Save the image
        if self.save_images and current_image is not None:
            self._save_image(current_image, step=self._timestep)
        
        # Determine if we should call VLM this step
        if self._should_call_vlm():
            # Call VLM to compute reward
            if self.use_subgoals:
                result = self.reward_interface.compute_reward_with_subgoals(
                    scene_image=current_image,
                    auto_advance=True,
                    completion_bonus=10.0
                )
                reward = result['reward']
                info['omni_reward_info'] = result
                
                # Cache the result
                self._last_vlm_reward = reward
                self._last_vlm_result = result
                self._steps_since_vlm_call = 0
                
                if result.get('all_completed', False):
                    terminated = True
            else:
                reward = self.reward_interface.compute_reward(current_image)
                self._last_vlm_reward = reward
                self._steps_since_vlm_call = 0
            
            info['vlm_called'] = True
        else:
            # Use interpolated/cached reward
            reward = self._compute_interpolated_reward()
            info['vlm_called'] = False
            
            # Still include last VLM result info if available
            if self._last_vlm_result is not None:
                info['omni_reward_info'] = self._last_vlm_result
            
        return obs, reward, terminated, truncated, info