"""Prompt templates for VLM caption generation."""

# Simple template - original (too brief)
SIMPLE_TEMPLATE = "Describe this image briefly."

# Structured template v1 - more detailed
STRUCTURED_V1_TEMPLATE = """Analyze this robotic manipulation scene in detail. Provide a comprehensive description including:

1. **Robot State**: 
   - End-effector/gripper position (approximate x, y, z coordinates or relative position)
   - Gripper state (open/closed, grasping something?)
   - Arm configuration and pose

2. **Objects in Scene**:
   - List all visible objects with their colors, shapes, and sizes
   - Position of each object (on table, in gripper, etc.)
   - Any notable features or markings

3. **Spatial Relationships**:
   - Distance between robot end-effector and objects
   - Relative positions (left/right, front/back, above/below)
   - Contact or near-contact situations

4. **Scene Context**:
   - Table/workspace description
   - Goal markers or target positions (if visible)
   - Any obstacles or constraints

5. **Current Action State**:
   - What action appears to be in progress?
   - Is the robot approaching, grasping, pushing, or placing?

Provide a detailed, structured description that captures all visual information relevant to understanding the robot's current state and progress toward any goal."""

# Goal-oriented template - focuses on goal progress
GOAL_ORIENTED_TEMPLATE = """You are analyzing a robotic manipulation scene. The robot's goal is: {goal}

Provide a detailed description of the current state including:

1. **Robot Configuration**:
   - Gripper/end-effector exact position and orientation
   - Whether gripper is open, closed, or grasping an object
   - Arm joint configuration if visible

2. **Task-Relevant Objects**:
   - Identify all objects relevant to the goal
   - Precise position of each object (use coordinates like "center-left of table", "near the red marker")
   - Object states (stationary, being moved, in gripper)

3. **Goal Progress Assessment**:
   - How close is the current state to the goal?
   - What has been accomplished so far?
   - What remains to be done?

4. **Spatial Details**:
   - Distances between key elements (robot to object, object to goal)
   - Alignments and orientations
   - Any obstacles between current state and goal

5. **Critical Observations**:
   - Any issues or challenges visible
   - Contact points or interactions
   - Environmental constraints

Be as specific and quantitative as possible in your description."""

# Detailed state template - maximum detail
DETAILED_STATE_TEMPLATE = """Provide an extremely detailed analysis of this robotic manipulation scene. 

**ROBOT ANALYSIS:**
- Describe the robot arm type and configuration
- End-effector position: specify approximate coordinates or use reference points
- End-effector orientation: which direction is it facing/pointing?
- Gripper state: fully open, partially open, closed, or grasping?
- If grasping, describe what is being held and how securely

**OBJECT INVENTORY:**
For EACH visible object:
- Object name and type
- Color and material appearance
- Size (small/medium/large or approximate dimensions)
- Exact position on the workspace
- Current state (stationary, in motion, in gripper, at goal)

**SPATIAL MAPPING:**
- Describe the workspace/table surface
- Map out object positions relative to each other
- Note any goal markers, target zones, or destination areas
- Identify any obstacles or constraints

**INTERACTION STATE:**
- Is the robot touching any object?
- Is any object being pushed, lifted, or manipulated?
- Describe any ongoing motion or action

**QUANTITATIVE ESTIMATES:**
- Estimate distances between key elements
- Describe positions using a coordinate system (e.g., "object is at table center, robot is 10cm to the left")
- Note heights above table surface

Provide maximum detail for accurate state representation."""

# Template dictionary for easy access
CAPTION_TEMPLATES = {
    "simple": SIMPLE_TEMPLATE,
    "structured_v1": STRUCTURED_V1_TEMPLATE,
    "goal_oriented": GOAL_ORIENTED_TEMPLATE,
    "detailed_state": DETAILED_STATE_TEMPLATE,
}

def get_template(template_name: str, goal: str = None) -> str:
    """Get a caption template by name, optionally formatted with goal."""
    template = CAPTION_TEMPLATES.get(template_name, STRUCTURED_V1_TEMPLATE)
    
    if goal and "{goal}" in template:
        return template.format(goal=goal)
    return template
