"""
After this generate async function connection string
Else generate node for each async node instantiated
"""


"""
And then weh search agent
And local agent
Then the AI thingy, add an OCR agent and map agent
"""

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.behaviors import DragBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.widget import Widget

lines = {}

connections = {}

class MousePositionWidget(Widget):
    def __init__(self, **kwargs):
        super(MousePositionWidget, self).__init__(**kwargs)
        self.prev_pos = None
        #self.total_dx = 0  # Total delta x
        #self.total_dy = 0  # Total delta y

    def on_touch_move(self, touch):
        if self.prev_pos:
            dx = touch.pos[0] - self.prev_pos[0]
            dy = touch.pos[1] - self.prev_pos[1]
            print("Mouse delta:", (dx, dy))

            if not any(isinstance(child, DraggableLabel) and child.dragging for child in self.parent.children):
                # If no box is being dragged, update total delta
                #self.total_dx += dx
                #self.total_dy += dy
                # Move all boxes by the total delta
                for child in self.parent.children:
                    if isinstance(child, DraggableLabel):
                        child.pos = (child.pos[0] + dx, child.pos[1] + dy)

        self.prev_pos = touch.pos
        return super(MousePositionWidget, self).on_touch_move(touch)
    def on_touch_up(self, touch):
        self.prev_pos = None


class TruncatedLabel(Label):
    def on_texture_size(self, instance, size):
        if size[0] > self.width:
            # Calculate the approximate width of "..." based on font size
            ellipsis_width = len("...") * self.font_size / 3
            # Calculate the maximum text width based on Label width and ellipsis width
            max_width = self.width - ellipsis_width
            text = self.text
            self.max_length = 7  # Maximum length of the text
            # Truncate the text to fit within the Label width
            if len(self.text) > self.max_length:
                self.text = self.text[:self.max_length] + "..."
            else:
                self.text = self.text
            """  
            while self.texture_size[0] > max_width and len(text) > 0:
                text = text[:-1]
                self.text = text + "..."
            """
        else:
            # Text fits within the Label width
            self.text = self.text

class DraggableLabel(DragBehavior, Label):
    def __init__(self, mouse_widget, **kwargs):
        super(DraggableLabel, self).__init__(**kwargs)
        self.text = "test"
        self.node_id = None
        self.size_hint = (None, None)
        self.size_x = 200
        self.size = (self.size_x, 50)
        #self.mouse_widget = mouse_widget  # Reference to the MousePositionWidget
        self.prev_pos = None  # Previous position of the widget
        self.dragging = False  # Flag to track whether the label is being dragged

        with self.canvas.before:
            self.input_labels = {}
            self.input_label_circles = {}
            
            self.output_labels = {}
            self.output_label_circles = {}
            
            self.line = None  # Initialize the line object
            self.line2 = None
            self.label_color = Color(0.5, 0.5, 0.5, 1)  # Set the label background color (gray in this case)
            self.label_rect = Rectangle(pos=self.pos, size=self.size)
            self.box_color = Color(0.3, 0.3, 0.3, 1)  # Set the box background color
            self.box_rect = Rectangle(pos=self.pos, size=self.size)

            # Define the positions of the input and output circles
            self.input_circle_pos = (self.x - 3, self.y + self.height / 2 - 5)
            self.output_circle_pos = (self.right - 7, self.y + self.height / 2 - 5)

            # Draw the input and output circles
            self.input_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.output_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.input_circle = Ellipse(pos=self.input_circle_pos, size=(10, 10))
            self.output_circle = Ellipse(pos=self.output_circle_pos, size=(10, 10))
            
            for i in range(1,3):
                # Add labels to the bottom box
                label = TruncatedLabel(text=f'Input {i}', size=(dp(len(f'Label {i}')*10), dp(10)))
                self.add_widget(label)
                self.input_labels[i] = label
                print(len(f'{i}')*10)
                print(self.input_labels[i])
                self.input_labels[i].pos = (self.x-15, self.y - self.height - (20 * i))
                
                ellipse_pos = (self.x-22, self.y - self.height - (20 * i))
                ellipse = Ellipse(pos=ellipse_pos, size=(10, 10))
                self.input_label_circles[i] = ellipse
            
            for i in range(1,2):
                # Add labels to the bottom box
                label = TruncatedLabel(text=f'Output {i}', size=(dp(len(f'Label {i}')*10), dp(10)))
                self.add_widget(label)
                self.output_labels[i] = label
                print(self.output_labels[i])
                self.output_labels[i].pos = (self.x-10, self.y - self.height - (20 * i))
                
                ellipse_pos = (self.x+2, self.y - self.height - (20 * i))
                ellipse = Ellipse(pos=ellipse_pos, size=(10, 10))
                self.output_label_circles[i] = ellipse

        self.bind(pos=self.update_rect, size=self.update_rect)
    
    def update_rect(self, *args):
        self.label_rect.pos = self.pos
        self.label_rect.size = self.size
        
        self.box_rect.pos = (self.x, self.y - self.height)
        self.box_rect.size = (self.width, self.height)
        # Update the positions of the input and output circles
        self.input_circle_pos = (self.x - 3, self.y + self.height / 2 - 5)
        self.output_circle_pos = (self.right - 7, self.y + self.height / 2 - 5)
        self.input_circle.pos = self.input_circle_pos
        self.output_circle.pos = self.output_circle_pos
        
        
        # Position the labels in the bottom box
        #self.label1.pos = (self.x-15, self.y - self.height - 15)
        #self.label2.pos = (self.x-15, self.y - self.height - 35)
        
        for i in range(1,3):
            self.input_labels[i].pos = (self.x-3, self.y - (20 * i))
            self.input_label_circles[i].pos = (self.x-3, self.y - (20 * i))
        
        for i in range(1,2):
            self.output_labels[i].pos = (self.x + self.width - self.output_labels[i].width, self.y - (20 * i))
            self.output_label_circles[i].pos = (self.x + self.width-7, self.y - (20 * i))
            
        if self.line2:
            self.line2.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                self.connection[0], self.connection[1]]
        #Loop through all connections of the name of this node and update the lines based on position of all circles in this nodes and the position of the connected circles
        #Loop through each connection
        #Get the position of the
    def on_touch_down(self, touch):
        
        if self.collide_point(*touch.pos):
            self.dragging = True  # Set dragging to True when touch is on the label
            # Check if the touch is within the bounds of the circles
            if (self.input_circle_pos[0] <= touch.pos[0] <= self.input_circle_pos[0] + 10 and
                    self.input_circle_pos[1] <= touch.pos[1] <= self.input_circle_pos[1] + 10) or \
               (self.output_circle_pos[0] <= touch.pos[0] <= self.output_circle_pos[0] + 10 and
                    self.output_circle_pos[1] <= touch.pos[1] <= self.output_circle_pos[1] + 10):
                # Change the circle color when held
                self.input_circle_color.rgba = (1, 0, 0, 1)  # Red color
                self.output_circle_color.rgba = (1, 0, 0, 1)  # Red color
                # Create a line from circle to touch position
                with self.canvas:
                    Color(1, 0, 0)
                    self.line = Line(points=[self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, *touch.pos])
                return super(DraggableLabel, self).on_touch_down(touch)
        # Allow dragging of the box
        self.drag_rectangle = (self.x, self.y, self.width, self.height)
        return super(DraggableLabel, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.line:
            self.line.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, *touch.pos]
        """
        if self.prev_pos:
            # Calculate the delta between the current and previous positions
            dx = touch.pos[0] - self.prev_pos[0]
            dy = touch.pos[1] - self.prev_pos[1]
            print("Delta from MousePositionWidget:", (dx, dy))
        self.prev_pos = touch.pos
        """
        
        return super(DraggableLabel, self).on_touch_move(touch)

    def is_point_in_ellipse(self, point, center, size):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        a = size[0]
        b = size[1]
        return (dx*dx) / (a*a) + (dy*dy) / (b*b) <= 1

    def on_touch_up(self, touch):
        self.dragging = False  # Set dragging to True when touch is on the label
        if self.line:
            # Check if any other DraggableLabel instances are colliding with the output circle
            for child in self.parent.children:
                if isinstance(child, DraggableLabel) and child != self:
                    #Make for loop to loop through all other nodes
                    #Check if collides with their box (Optional)
                    #Loop through every input in that node
                    """
                    for input_circle in child.input_label_circles:
                        print(input_circle.pos)
                        #Print key and value
                    """
                    
                    if self.is_point_in_ellipse(touch.pos, child.input_circle_pos, (10, 10)):
                        # Create a line connecting the output circle of the current instance to the input circle of the other instance
                        with self.canvas:
                            Color(1, 0, 0)
                            self.line2 = Line(points=[self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                                      child.input_circle_pos[0] + 5, child.input_circle_pos[1] + 5])
                            self.connection = (child.input_circle_pos[0] + 5, child.input_circle_pos[1] + 5)
                            #If collided save the id of the connection bidirectionally, the id of this and the other.
                            #First get the id of the child, store in connections
                            print(child.text)
                            #The circle collided in that id, store in connections[id]
                            
                            #The id of this node, store in connections
                            #The id of circle of this node, from which is detected from touch_down store in self
                            
                            #Create a line globally
                            
                            
                            print(self.connection)
                            print(child.text)
                        break
            self.canvas.remove(self.line)
            self.line = None
            self.input_circle_color.rgba = (1, 1, 1, 1)  # Gray color
            self.output_circle_color.rgba = (1, 1, 1, 1)  # Gray color
        return super(DraggableLabel, self).on_touch_up(touch)


    def clear_canvas(self):
        self.canvas.before.clear()
        with self.canvas.before:
            self.label_color = Color(0.5, 0.5, 0.5, 1)
            self.label_rect = Rectangle(pos=self.pos, size=self.size)
            self.box_color = Color(0.3, 0.3, 0.3, 1)
            self.box_rect = Rectangle(pos=(self.x, self.y - self.height), size=(self.width, self.height))
            self.input_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.output_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.input_circle = Ellipse(pos=self.input_circle_pos, size=(10, 10))
            self.output_circle = Ellipse(pos=self.output_circle_pos, size=(10, 10))

node = {
    "name" : "add",
    "func" : None,
    "inputs" : {
        "a" : "num", 
        "b" : "num"
    },
    "outputs": {
        "c" : "num"
    }
}

#def newNode(node):
    
    
class DraggableLabelApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        mouse_widget = MousePositionWidget()
        
        draggable_label1 = DraggableLabel(mouse_widget=mouse_widget)
        draggable_label2 = DraggableLabel(mouse_widget=mouse_widget)
        
        
        
        layout.add_widget(mouse_widget)
        layout.add_widget(draggable_label1)
        layout.add_widget(draggable_label2)

        return layout

if __name__ == '__main__':
    DraggableLabelApp().run()
