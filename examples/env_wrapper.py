'''
this is Environment wrapper that replaces rewards with OmniReward
connect OmniReward with your task environment
'''
# import gym
import os
import sys
import logging
from datetime import datetime
import numpy as np
import numpy.typing as npt
from PIL import Image
from omni_reward.reward.interface import OmniRewardInterface


def setup_logger(task_name: str, log_dir: str = "./logs") -> logging.Logger:
    """Setup logger that writes to both console and file.
    
    Args:
        task_name: Name of the task for log file naming
        log_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp for unique log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{task_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger(f"OmniReward_{task_name}_{timestamp}")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler - writes to file
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - prints to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file created: {log_filepath}")
    
    return logger


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
        # Logging settings
        save_logs: bool = True,
        log_dir: str = "./logs",
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
        date_str = datetime.now().strftime("%Y%m%d")
        self.image_save_dir = image_save_dir if image_save_dir else f"./images/{task_name}/{date_str}"
        if self.save_images:
            os.makedirs(self.image_save_dir, exist_ok=True)
        
        # Logging settings
        self.save_logs = save_logs
        self.log_dir = log_dir
        if self.save_logs:
            self.logger = setup_logger(task_name, log_dir)
            self._log_init_info()
        else:
            self.logger = None
        
        self._timestep = 0
        self._episode = 0
    
    def _log_init_info(self):
        """Log initialization information"""
        self.logger.info("=" * 80)
        self.logger.info("OmniRewardWrapper Initialized")
        self.logger.info("=" * 80)
        self.logger.info(f"Task Name: {self.task_name}")
        self.logger.info(f"Goal Text: {self.goal_text}")
        self.logger.info(f"Use Subgoals: {self.use_subgoals}")
        self.logger.info(f"Camera: {self.camera_name}")
        self.logger.info(f"VLM Call Interval: {self.vlm_call_interval}")
        self.logger.info(f"Image Save Dir: {self.image_save_dir}")
        self.logger.info("=" * 80)
    
    def _log(self, message: str, level: str = "info"):
        """Log a message if logging is enabled"""
        if self.logger:
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(message)
        else:
            print(message)
    
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
                        self._log(f"Camera {cam} not found, using default: {e}", "warning")
                        renderer.update_scene(data)
                
                img = renderer.render()
                renderer.close()
                
                # mujoco renders images upside down, need to flip
                img = img[::-1]
                return img
            except Exception as e:
                self._log(f"mujoco_renderer failed for camera {cam}: {e}", "error")
        
        if hasattr(self.env, 'sim'):
            # Older MetaWorld uses sim (mujoco_py)
            try:
                width = self.env.width if hasattr(self.env, 'width') else 480
                height = self.env.height if hasattr(self.env, 'height') else 480
                
                if cam == "frontview":
                    self._log("Warning: frontview not supported with mujoco_py, using corner", "warning")
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
                self._log(f"sim.render failed for camera {cam}: {e}", "error")
        
        # Finally, try calling env.render()
        self._log(f"Warning: Using default render(), camera '{cam}' may not be applied", "warning")
        return self.env.render()
    
    def _get_multi_view_images(self) -> dict:
        """Get images from multiple camera views"""
        cameras = ["corner", "corner2", "corner3", "topview", "behindGripper"]
        images = {}
        for cam in cameras:
            try:
                images[cam] = self._get_image(camera_name=cam)
            except Exception as e:
                self._log(f"Failed to render camera {cam}: {e}", "error")
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
        
        self._log(f"\n{'='*80}")
        self._log(f"Episode {self._episode} Started")
        self._log(f"{'='*80}")
        
        # Reset VLM call tracking
        self._steps_since_vlm_call = self.vlm_call_interval  # Force VLM call on first step
        self._last_vlm_reward = 0.0
        self._last_vlm_result = None
        
        # NOTE: Do not call reset_episode() here, as start_episode_with_subgoals will handle it.
    
        # Get the initial image (using the specified camera)
        initial_image = self._get_image(camera_name=self.camera_name)
        
        # Save the initial image
        if self.save_images and initial_image is not None:
            self._save_image(initial_image, step=0)
        
        if self.use_subgoals:
            # start_episode_with_subgoals will set goal_text and subgoals
            self.reward_interface.start_episode_with_subgoals(
                self.goal_text, 
                initial_image=initial_image,
                auto_decompose=True
            )
            # Log subgoals
            if hasattr(self.reward_interface, 'subgoals') and self.reward_interface.subgoals:
                self._log(f"Subgoals ({len(self.reward_interface.subgoals)}):")
                for i, sg in enumerate(self.reward_interface.subgoals):
                    self._log(f"  {i+1}. {sg}")
        else:
            # start_episode will set goal_text
            self.reward_interface.start_episode(self.goal_text, initial_image=initial_image)
        
        self._log(f"Initial observation shape: {obs.shape}")
        
        return obs, info
    
    def step(self, action):
        # obs, _, done, info = self.env.step(action) # NOTE:old version < gym 0.26 does not return info
        obs, _, terminated, truncated, info = self.env.step(action) # gym 0.26+
        done = terminated or truncated
        if self.use_subgoals:
            # Compute reward using subgoal-based progression
            result = self.reward_interface.compute_reward_with_subgoals(
                scene_image=obs, #TODO: check if obs contains image or need to extract
                timestep=self.reward_interface.timestep,
                auto_advance=True,
                completion_bonus=10.0  # Extra reward for completing a subgoal
            )
            
            reward = result['reward']
            info['omni_reward_info'] = result
            
            info['vlm_called'] = True
        else:
            # Compute reward directly from the final goal
            reward = self.reward_interface.compute_reward(obs, timestep=self.reward_interface.timestep, goal_text=self.goal_text
                                                          )
            
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment and finalize logging"""
        if self.logger:
            self._log(f"\n{'='*80}")
            self._log(f"Environment Closed")
            self._log(f"Total Episodes: {self._episode}")
            self._log(f"{'='*80}")
            
            # Close all handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        self.env.close()