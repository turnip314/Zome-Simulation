import simplegui
import math

PHI = (1+5**0.5)/2
SCALE = 170
SHIFT = SCALE*2
AUTO_TIME = 1000

#USER CONTROL STUFF

spin = False
fixed = False
cross_data_collecting = True
visible = True
fix_message = "Loosened"
collect_message = "Collecting points"
visible_message = "Shape visible"
drag_pos = (0,0,0)
points = []
lines = []
overlapped = []

ep = 0.04

auto_generate = 0

# global direction vectors
i = [1,0,0]
j = [0,1,0]

# INPUT HANDLERS
# Handler for mouse click
def spin_handler():
    global spin, fixed, fix_message
    spin = not spin
    if spin:
        fixed = True
        fix_message = "Fixed to Axis"
    
def fix_handler():
    global fixed, fix_message
    if not spin:
        fixed = not fixed
        if fixed:
            fix_message = "Fixed to Axis"
        else:
            fix_message = "Loosened"
        
def intersect_handler():
    global cross_data_collecting, collect_message
    cross_data_collecting = not cross_data_collecting
    if cross_data_collecting:
        collect_message = "Collecting points"
    else:
        collect_message = "Not collecting points"
    
def clear_handler():
    global points, lines
    points = []
    lines = []
    
def visible_handler():
    global visible, visible_message
    visible = not visible
    if visible:
        visible_message = "Shape visible"
    else:
        visible_message = "Shape hidden"
        
def auto_handler():
    global auto_generate
    auto_generate = AUTO_TIME
    if not fixed:
        fix_handler()
    
    
def drag(pos): # 2d rotation matrices
    if(not fixed and auto_generate <= 0):
        global drag_pos

        vert = (pos[1] - drag_pos[1])/100.0
        if (math.fabs(vert) < 0.2):
            for vtx in Object.vtxs:
                y, z = vtx[1], vtx[2]
                vtx[1] = y*math.cos(vert) - z*math.sin(vert)
                vtx[2] = y*math.sin(vert) + z*math.cos(vert)

        hor = (pos[0] - drag_pos[0])/100.0
        if (math.fabs(hor) < 0.4):
            for vtx in Object.vtxs:
                x, z = vtx[0], vtx[2]
                vtx[0] = x*math.cos(hor) - z*math.sin(hor)
                vtx[2] = x*math.sin(hor) + z*math.cos(hor)
    elif (not spin and auto_generate <= 0):
        drag_direc = [pos[0] - drag_pos[0], 
                      pos[1] - drag_pos[1]]
        
        if  mag(drag_direc) < 20:
            axis_direc = [Object.get_axis()[0][0] - Object.get_axis()[1][0],
                      Object.get_axis()[0][1] - Object.get_axis()[1][1]]
            
            amount = cross(axis_direc, drag_direc)/(100*mag(axis_direc))
            Object.spin(amount)
            
        
    drag_pos = pos
# shape; axis should pass through origin
class Shape:
    def __init__(self, vtxs, edges, axis):
        self.vtxs = vtxs
        self.edges = edges
        self.lines = [] #pair of vertices
        self.axis = axis
    def draw(self, canvas):
        if not visible:
            return
        self.lines = []
        for e in self.edges:
            self.lines.append([self.vtxs[e[0]],
                              self.vtxs[e[1]]])
        for l in self.lines:
            draw_edge(canvas, l)
        # draw axis
        canvas.draw_line(proj(self.get_axis()[0],i,j),
                         proj(self.get_axis()[1],i,j),
                         4, 'Green')
        for p in self.axis:
            canvas.draw_circle(proj(self.vtxs[p],i,j),
                               SCALE/10.0, 1, 'Green', 'Green')
        for vtx in self.vtxs:
            draw_coords(canvas, vtx)
    def get_axis(self):
        return [self.vtxs[self.axis[0]],
                self.vtxs[self.axis[1]]]
    
    def spin(self, speed):
        v = [[],[],[]] # for conv matrix
        for k in range(3):
            v[2].append(self.get_axis()[1][k]-
                        self.get_axis()[0][k])
        v[0] = [-1*v[2][1], v[2][0], 0]
        v[1] = [-1*v[2][0]*v[2][2], -1*v[2][1]*v[2][2],
                v[2][0]**2+v[2][1]**2]
        # basis changes (z -> axis vector)
        conv = Matrix(3,3,v).trans().norm()
        rot_orig = Matrix(3,3,[[math.cos(speed),
                                 -math.sin(speed),0],
                                [math.sin(speed),
                                 math.cos(speed),0],
                                [0,0,1]])
        # applying conjugation
        rot = conv.times(rot_orig.times(conv.trans()))
        
        for vtx in self.vtxs: # rotates vtx in 3d w/ matrix
            mat = Matrix(3,1,[[vtx[0]],[vtx[1]],[vtx[2]]])
            rotated = rot.times(mat)
            for k in range(3):
                vtx[k] = rotated.nums[k][0]
                
        if cross_data_collecting:
            for l in self.lines:
                for m in self.lines:
                    if l != m and [l,m] not in overlapped:
                        v1 = [collapse(l[0]), collapse(l[1])]
                        v2 = [collapse(m[0]), collapse(m[1])]
                        
                        overlap = get_overlap(v1, v2)
                        if overlap != None:
                            overlapped.append([l,m])
                            overlapped.append([m,l])
                            line = [[SHIFT+SCALE*overlap[0][0],SHIFT+SCALE*overlap[0][1]], 
                                              [SHIFT+SCALE*overlap[1][0],SHIFT+SCALE*overlap[1][1]]]
                            if is_new_line(line):
                                lines.append(line)
                        
                        intersect = get_intersect(get_eq(v1[0], v1[1]), get_eq(v2[0], v2[1]))
                        if does_intersect(v1, v2, intersect):
                            #print self.lines.index(l), self.lines.index(m)
                            new_p = [SHIFT+SCALE*intersect[0], SHIFT+SCALE*intersect[1]]
                            #print len(points)
                            #print new_p
                            append = True
                            """for p in points:
                                #print mag([p[0]-new_p[0], p[1]-new_p[1]])
                                if mag([p[0]-new_p[0], p[1]-new_p[1]]) < 10:
                                    append = False
                            for vtx in self.vtxs:
                                if mag([proj(vtx,i,j)[0]-new_p[0], proj(vtx,i,j)[1]-new_p[1]]) < 0.01:
                                    append = False"""
                            if append:
                                points.append(new_p)
                                
                        else:
                            pass
                            #print intersect
                            #print v1, v2, intersect
             
#DRAWINGS AND SHAPE RELATED STUFF
object_vtxs= {"cube": [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],
            [1,-1,1],[1,1,-1],[1,1,1]],
              "tetrahedron": [[0,0,1],[8**0.5/3,0,-1*(1/3)],[-1*2**0.5/3,2**0.5/3**0.5,-1*(1/3)],[-1*2**0.5/3,-1*2**0.5/3**0.5,-1*(1/3)],[0,0,-1*(1/3)]],
              "octahedron": [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
              "icosahedron": [[0,1,PHI],[0,-1,PHI],[0,1,-1*PHI],[0,-1,-1*PHI],
                              [PHI,0,1],[PHI,0,-1],[-1*PHI,0,1],[-1*PHI,0,-1],
                              [1,PHI,0],[-1,PHI,0],[1,-1*PHI,0],[-1,-1*PHI,0]],
              "dodecahedron":[[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],
                               [-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,PHI,1/PHI],[0,PHI,-1*(1/PHI)],
                               [0,-1*PHI,1/PHI],[0,-1*PHI,-1*(1/PHI)],[1/PHI,0,PHI],
                               [1/PHI,0,-1*PHI],[-1*(1/PHI),0,PHI],[-1*(1/PHI),0,-1*PHI],
                               [PHI,1/PHI,0],[PHI,-1*(1/PHI),0],[-1*PHI,1/PHI,0],
                               [-1*PHI,-1*(1/PHI),0]],
              "tetracube":[[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],
            [1,-1,1],[1,1,-1],[1,1,1]]
              }
object_edges= {"cube":[[0,1],[1,3],[3,2],[2,0],[0,4],[1,5],[2,6],
                  [3,7],[4,5],[5,7],[7,6],[6,4]],
               "tetrahedron":[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],
               "octahedron" :[[4,0],[4,1],[4,2],[4,3],[5,0],[5,1],[5,2],[5,3],[0,2],[2,1],[1,3],[3,0]],
               "icosahedron":[[0,1],[0,6],[0,4],[0,8],[0,9],[1,4],[1,6],[1,11],
                              [1,10],[4,5],[4,8],[4,10],[2,3], [2,5],[2,7], [2,8],
                              [2,9],[3,5],[3,7],[3,10],[3,11],[5,8],[5,10],[6,7],
                              [6,9],[6,11],[7,9],[7,11],[8,9],[10,11]],
               "dodecahedron":[[0,12],[0,16],[0,8],[1,13],[1,9],[1,16],[2,12],
                               [2,10],[2,17],[3,13],[3,11],[3,17],[4,8],
                               [4,14],[4,18],[5,15],[5,18],[5,9],[6,14],
                               [6,19],[6,10],[7,19],[7,15],[7,11],[8,9],
                               [10,11],[12,14],[13,15],[16,17],[18,19]],
               "tetracube":[[0,1],[1,3],[3,2],[2,0],[0,4],[1,5],[2,6],
                  [3,7],[4,5],[5,7],[7,6],[6,4],[1,7],[1,4],[4,7],
                            [1,2],[2,7],[2,4]]
               }
object_axis = {"cube":[0,7],
               "tetrahedron": [0,4],
               "octahedron": [4,5],
               "icosahedron": [0,3],
               "dodecahedron":[12,15],
               "tetracube":[1,6]
              }
def input_handler(user_input):
    global Object
    if object_vtxs.has_key(user_input):
        Object = Shape(object_vtxs[user_input], 
                       object_edges[user_input], 
                       object_axis[user_input])
    else:
        return None

# Handler to draw on canvas
def draw(canvas):
    global auto_generate
    Object.draw(canvas)
    if spin:
        Object.spin(0.1)
    canvas.draw_text(fix_message, [0,15], 14, "white")
    canvas.draw_text(collect_message, [0,30], 14, "white")
    canvas.draw_text(visible_message, [0,45], 14, "white")
    for point in points:
        canvas.draw_point(point, "yellow")
    for line in lines:
        canvas.draw_line(line[0], line[1], 2, "yellow")
        
    if auto_generate > 0:
        Object.spin(2*math.pi/AUTO_TIME)
        auto_generate -= 1
        canvas.draw_line([0,4*SCALE],[4*SCALE*auto_generate/AUTO_TIME, 4*SCALE], 20, "green")

# draw edge
def draw_edge(canvas, l):
    canvas.draw_line(proj(l[0],i,j),proj(l[1],i,j),
                     4, 'blue')
    
# draw coordinates
def draw_coords(canvas, vtx):
    text = "(" + str(cut(vtx[0], 2)) + "," + str(cut(vtx[1], 2)) + "," + str(cut(vtx[2], 2)) + ")"
    canvas.draw_text(text, [SHIFT + SCALE*vtx[0],
                           SHIFT + SCALE*vtx[1]], 12, "white")



#MATH STUFF HERE

# truncates decimal at certain point x
def cut(num, x):
    return math.trunc(num*10**x)/10.0**x

# cross product (2D only, because other dimensions are too hard ;-;)
def cross(v,w):
    return v[0]*w[1] - v[1]*w[0]

# returns positive angle between two lines
def angle_var(l1,l2):
    v1 = [l1[0][0]-l1[1][0], l1[0][1]-l1[1][1]]
    v2 = [l2[0][0]-l2[1][0], l2[0][1]-l2[1][1]]
    if mag(v1)*mag(v2) < ep:
        pass
        #print v1, v2
    return math.asin(math.fabs(cross(v1, v2)/
                               (mag(v1)*mag(v2))))

# dot product (any dimension)
def dot(v,w):
    sum = 0
    for i in range(len(v)):
        sum += v[i]*w[i]
    return sum

# magnitude (any dimension)
def mag(v):
    return math.sqrt(dot(v,v))

# returns only the x and y coordinates of a vector
def collapse(v):
    return [v[0], v[1]]

# projection v onto plane with x,y (usually i,j)
def proj(v,x,y):
    return(SHIFT+SCALE*dot(v,x)/mag(x),
           SHIFT+SCALE*dot(v,y)/mag(y));

# returns [m, b] of the form y = mx + b
def get_eq(v, w):
    if w[0] != v[0]:
        m = (w[1] - v[1])/(w[0] - v[0])
        b = -m*v[0] + v[1]
        return [m, b]
    else:
        return [1000, 1000]

# gets endpoints of two segments and if they overlap, returns the endpoints of the segment created by the overlap
def get_overlap(m,n):
    m.sort()
    n.sort()
    eq1=get_eq(m[0],m[1])
    eq2=get_eq(n[0],n[1])
    #print angle_var(m,n), 1
    close_enough = False
    if angle_var(m,n) > 0.01*ep and angle_var(m,n) < ep:
        #print m
        #print n
        point = get_intersect(eq1, eq2)
        center = [0.25*(m[0][0]+m[1][0]+n[0][0]+n[1][0]),
                  0.25*(m[0][1]+m[1][1]+n[0][1]+n[1][1])]
        if mag([center[0]-point[0],center[1]-point[1]]) < ep/math.sin(angle_var(m,n)):
            close_enough = True
        #print close_enough
    if math.fabs(eq1[1]-eq2[1]) < ep/math.sin(angle_var(m,[[0,0],[0,1]])) and angle_var(m,n) < ep:
        close_enough = True
    if close_enough:
        #print angle_var(m,n), 2
        if (n[0][0]<m[0][0]<n[1][0] and n[0][0]<m[1][0]<n[1][0]):
            return m
        elif (m[0][0]<n[0][0]<m[1][0] and m[0][0]<n[1][0]<m[1][0]):
            return n
        elif (m[0][0]<n[0][0]<m[1][0]):
            return [[n[0][0],n[0][1]],[m[1][0],m[1][1]]]
        elif m[0][0]<n[1][0]<m[1][0]:
            return [[m[0][0],m[0][1]],[n[1][0],n[1][1]]]
        
# find the points of intersection of two equations, gives [x,y]
# will return jank if slope is undefined 
def get_intersect(eq1, eq2):
    #print eq1, eq2
    if eq1[0] - eq2[0] != 0:
        x_sol = (eq2[1]-eq1[1])/(eq1[0]-eq2[0])
        y_sol = eq1[0]*x_sol + eq1[1]
        return [x_sol, y_sol]
    else:
        return [-1000, -1000]

# checks if the two line segments actually intersect in supposed region from above
def does_intersect(l1, l2, sol):
    #return True
    return (l1[0][0] <= sol[0] <= l1[1][0] or l1[0][0] >= sol[0] >= l1[1][0]) and \
           (l1[0][1] <= sol[1] <= l1[1][1] or l1[0][1] >= sol[1] >= l1[1][1]) and \
           (l2[0][0] <= sol[0] <= l2[1][0] or l2[0][0] >= sol[0] >= l2[1][0]) and \
           (l2[0][1] <= sol[1] <= l2[1][1] or l2[0][1] >= sol[1] >= l2[1][1])
    

def is_new_line(l):
    l_eq = get_eq(l[0], l[1])
    for line in lines:
        line_eq = get_eq(line[0], line[1])
        if angle_var(l, line) < ep and math.fabs(l_eq[1] - line_eq[1]) < SCALE*ep:
            break
            return False
    return True
        
# matrix class; m rows, n columns; [[row],[row]]
class Matrix:
    def __init__(self,m,n,nums):
        self.m = m
        self.n = n
        self.nums = nums
    def trans(self): # transpose
        new_nums = []
        for k1 in range(self.n):
            row = []
            for k2 in range(self.m):
                row.append(self.nums[k2][k1])
            new_nums.append(row)
        new_matrix = Matrix(self.n, self.m, new_nums)
        return new_matrix
    def times(self, b): # matrix product; b(n*k) on right
        new_nums = []
        for k1 in range(self.m):
            row = []
            for k2 in range(b.n):
                element = 0
                for k3 in range(self.n):
                    element += self.nums[k1][k3]*b.nums[k3][k2]
                row.append(element)
            new_nums.append(row)
        new_matrix = Matrix(self.m, b.n, new_nums)
        return new_matrix
    def norm(self): # normalization of each column
        new_nums = []
        for k1 in range(self.m):
            row = []
            for k2 in range(self.n):
                row.append(self.nums[k1][k2])
            new_nums.append(row)
        for k2 in range(self.n):
            mag = 0
            for k1 in range(self.m):
                mag += new_nums[k1][k2] ** 2
            mag = mag ** 0.5
            for k1 in range(self.m):
                new_nums[k1][k2] /= mag
        new_matrix = Matrix(self.m, self.n, new_nums)
        return new_matrix
    def str(self):
        print self.m, self.n, self.nums

# Create a frame and assign callbacks to event handlers
frame = simplegui.create_frame("Home", 4*SCALE, 4*SCALE)
frame.add_input('Object', input_handler, 100)
frame.add_button("Spin", spin_handler)
frame.add_button("Fixed Axis", fix_handler)
frame.add_button("Mark Intersect", intersect_handler)
frame.add_button("Clear Points", clear_handler)
frame.add_button("Show/Hide Shape", visible_handler)
frame.add_button("Auto Generate", auto_handler)
frame.set_draw_handler(draw)
frame.set_mousedrag_handler(drag)
# Start the frame animation
input_handler("cube")
frame.start()

