
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import copy
import os
import re
import random
import timeit



file_path = "Dataset\\Selected_dataset\\"

save_path = "Results_UncertainKmeans\\"
if not os.path.exists(save_path):
    os.makedirs(save_path)

file_list = os.listdir(file_path)
# print(file_list[1][:-4])
file_extention = ".txt"


###############################################
####function to calculate manhattan distance####
###############################################
def manhattan_dist(p1, p2):
    tmp_x = abs(p1[0] - p2[0])
    tmp_y = abs(p1[1] - p2[1])
    return (tmp_x + tmp_y)


def which_part(p_x, p_y, x_err, y_err, c_s):
    a = p_x - x_err
    b = p_x + x_err
    c = p_y - y_err
    d = p_y + y_err
    c_s_x = c_s[0]
    c_s_y = c_s[1]
    if ((c_s_x >= a )&(c_s_x <= b)&(c_s_y >= c)&(c_s_y <= d)):
        return "r"
    else:
        if ((c_s_x >= a )&(c_s_x <= b)):
            return "1"
        else:
            if ((c_s_y >= c) & (c_s_y <= d)):
                return "2"
            else:
                return "3"

######################################
####function to create error boxes####
######################################

def make_error_boxes(ax, xdata, ydata, xerror, yerror, colour_lists, edgecolor='None', alpha=0.5):
    # facecolor = 'r'

    # Create list for all the error patches
    errorboxes = []
    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=colour_lists, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='None')

    return artists

##################################
####Compute ED(point1, point2)####
##################################

def compute_ed(p,c_s, x_err, y_err):
    case_id = which_part(p[0], p[1],x_err, y_err, c_s)
    # case_id = "3"
    a = p[0] - x_err
    b = p[0] + x_err
    c = p[1] - y_err
    d = p[1] + y_err
    if (case_id == "r"):
        return (abs(d-c)/2.0 + abs(b-a)/2.0)
    else:
        if (case_id == "1"):
            return (abs(b-a)/2.0 + ((d+c)/2.0) - c_s[1])
        else:
            if (case_id == "2"):
                return (abs(d-c)/2.0 + ((a+b)/2.0) - c_s[0])
            else:
                if (case_id == "3"):
                    return (abs(p[0]-c_s[0]) + abs(p[1]-c_s[1]))





number_of_cluster = {
    "a1" : 20,
    "a2" : 35,
    "a3" : 50,
    "dim2" : 9,
    "g2-2-10" : 2,
    "g2-2-20" : 2,
    "unbalance" : 8,
    "R15" : 15,
    "Aggregation" : 7,
    "MopsiLocations2012-Joensuu" : 40,
    "MopsiLocationsUntil2012-Finland" : 40
}




g = open(save_path + "result_" + re.sub('.py', '',os.path.basename(__file__))  +  ".txt", 'w')
g.write("FileName      \t   N   \t  k   \t    Avg_dist    \t  Runtime\n")

for filename in file_list:
    number_of_center = number_of_cluster[filename[:-4]]
    print("File Name: " + filename[:-4])
    image_save_path = "Results_UncertainKmeans\\Images_" + filename[:-4] + "\\"
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    mylist = list()
    x_point = list()
    y_point = list()
    x_y_points = list()
    pwr_counter = 0
    with open(file_path + filename , 'r') as f:
        for line in f:
            tmp_line = line.split()
            # mylist.append(tmp_line)
            tmp_list = list()
            tmp_list.append(float(tmp_line[0]))
            pwr_counter += 1
            tmp_list.append(float(tmp_line[1]))
            if (tmp_list not in x_y_points):
                x_y_points.append(tmp_list)
                x_point.append(float(tmp_line[0]))
                y_point.append(float(tmp_line[1]))


    f.close()
    candid_points_x = list()
    candid_points_y = list()
    candid_xy = list()



    max_x = max(x_point)
    min_x = min(x_point)
    max_y = max(y_point)
    min_y = min(y_point)
    x_error_range = (max_x - min_x) / 40.0
    y_error_range = (max_y - min_y) / 40.0




    number_of_points = len(x_y_points)
    print( "Number Of Points: " + str(number_of_points))
    print( "Number of Clusters: " + str(number_of_center))

    xerr = np.random.rand(2, number_of_points) * x_error_range
    x_error = list()
    x_error.extend(copy.copy(list(xerr[0])))

    yerr = np.random.rand(2, number_of_points) * y_error_range
    y_error = list()
    y_error.extend(copy.copy(list(yerr[0])))


    # tmp_err = np.random.rand(1, number_of_points) * 1.0
    # xerr = np.empty( number_of_points)
    # x_error = list()
    # x_error.extend(copy.copy(list(tmp_err[0])))
    # xerr = copy.copy(tmp_err[0])

    # tmp_err = np.random.rand(1, number_of_points) * 1.0
    # yerr = np.empty(number_of_points)
    # y_error = list()
    # y_error.extend(copy.copy(list(tmp_err[0])))
    # yerr = copy.copy(tmp_err[0])

    colors = ['b', 'g', 'r']
    colour_kmeans = ['#8B8378', '#458B74', '#838B8B', '#E3CF57', '#000000', '#0000FF', '#8A2BE2', '#9C661F','#FF4040', '#98F5FF', '#FF6103', '#7FFF00', '#3D59AB','#FFD700', '#228B22', '#1C86EE', '#FF1493', '#9400D3', '#97FFFF', '#C1FFC1', '#FF8C00', '#7CFC00', '#F08080', '#7A8B8B', '#FFB6C1', '#8B5F65', '#20B2AA', '#8470FF', '#87CEFA', '#32CD32', '#FF00FF', '#03A89E', '#800000', '#5D478B', '#7B68EE', '#C71585', '#191970', '#8B7D7B', '#C0FF3E', '#FF8000', '#FF4500', '#DA70D6', '#FFFF00', '#00C78C', '#8B7B8B', '#EE9A49', '#33A1C9', '#FFC125', '#EE6AA7', '#9F79EE', '#AEEEEE']
    colour = ['#8B8378', '#458B74', '#838B8B', '#E3CF57', '#000000', '#0000FF', '#8A2BE2', '#9C661F','#FF4040', '#98F5FF', '#FF6103', '#7FFF00', '#3D59AB','#FFD700', '#228B22', '#1C86EE', '#FF1493', '#9400D3', '#97FFFF', '#C1FFC1', '#FF8C00', '#7CFC00', '#F08080', '#7A8B8B', '#FFB6C1', '#8B5F65', '#20B2AA', '#8470FF', '#87CEFA', '#32CD32', '#FF00FF', '#03A89E', '#800000', '#5D478B', '#7B68EE', '#C71585', '#191970', '#8B7D7B', '#C0FF3E', '#FF8000', '#FF4500', '#DA70D6', '#FFFF00', '#00C78C', '#8B7B8B', '#EE9A49', '#33A1C9', '#FFC125', '#EE6AA7', '#9F79EE', '#AEEEEE']
    colour1 = ['b', 'g', 'r', 'm', 'y', 'c', 'k', 'w']
    colour_kmeans1 = ['b', 'g', 'r', 'm', 'y', 'c', 'k', 'w']
    markers = ['o', 'v', 's']
    marker1 = ["8", "s", "p", "P", "*", "h", "H", "+", "x", "o", "v"]
    marker = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd','|', '_']
    
    KK = number_of_center-1
    number_of_run = 5
    sum_dist = 0
    sum_obj = 0
    
    elapsed_cluster = 0.0
    for run_index in range(number_of_run):
        start_time_cluster = timeit.default_timer()
        print ("run: " + str(run_index + 1))
        objective_value = [0] * number_of_center
        distance_list = list()
        center = np.empty((2, number_of_center))
        c_list = list()
        rand_c = random.randint(0, number_of_points - 1)
        c_list.append(copy.copy(x_y_points[rand_c]))
        center[0][0] = copy.copy(x_point[rand_c])
        center[1][0] = copy.copy(y_point[rand_c])
        counter = 0
        ex_distance_matrix = list()
        for i in range(number_of_points):
            point1 = copy.copy(x_y_points[i])
            tmp_value = list()
            for j in range(number_of_points):
                point2 = copy.copy(x_y_points[j])
                tmp_value.append(compute_ed(point1, point2, x_error[i],y_error[i]))
            ex_distance_matrix.append(copy.copy(tmp_value))
                
                
        for kkk in range(number_of_center - 1):
            ed_value = list()
            expected_distance = list()
            for point_index in range(number_of_points):
                point = copy.copy(x_y_points[point_index])
                if (point not in c_list):
                    # tmp_value = list()
                    tmp_ed = list()
                    for c_index in range(len(c_list)):
                        c_i = c_list[c_index]
                        # tmp_value.append((compute_ed(point, c_i, x_error[point_index],y_error[point_index]),point_index))
                        tmp_ed.append((copy.copy(ex_distance_matrix[point_index][x_y_points.index(c_list[c_index])]),point_index))
                    # min_in_list = min(tmp_value, key=lambda x: x[0])
                    min_in_list = min(tmp_ed, key=lambda x: x[0])
                    ed_value.append([copy.copy(min_in_list[0]), copy.copy(min_in_list[1])])
            tmp_center = copy.copy(x_y_points[max(ed_value, key=lambda x: x[0])[1]])
            c_list.append(copy.copy(tmp_center))
            counter += 1
            center[0][counter] = copy.copy(tmp_center[0])
            center[1][counter] = copy.copy(tmp_center[1])


        colour_list = [-1] * number_of_points
        point_cluster_index = [-1] * number_of_points

        for point_index in range(number_of_points):
            point = copy.copy(x_y_points[point_index])
            if (point not in c_list):
                tmp_value = list()
                for c_index in range(len(c_list)):
                    c_i = c_list[c_index]
                    tmp_value.append((manhattan_dist(point, c_i),point_index))
                indice = tmp_value.index(min(tmp_value, key=lambda x: x[0]))
                colour_list[point_index] = colour[indice]
                point_cluster_index[point_index] = indice
            else:
                colour_list[point_index] = colour[c_list.index(point)]
                point_cluster_index[point_index] = c_list.index(point)
        
        elapsed_cluster += (timeit.default_timer() - start_time_cluster)

        count = 0
        for i in range(number_of_points):
            tmp_dist = manhattan_dist(x_y_points[i], c_list[point_cluster_index[point_index]])
            tmp_obj = (manhattan_dist(x_y_points[i], c_list[point_cluster_index[point_index]])) ** 2
            distance_list.append(copy.copy(tmp_dist))
            objective_value[point_cluster_index[point_index]] += tmp_obj
        max_distance = max(distance_list)
        max_objective = max(objective_value)
        sum_dist += max_distance
        sum_obj += max_objective

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Call function to create error boxes
        _ = make_error_boxes(ax, np.array(x_point), np.array(y_point), xerr, yerr, colour_list)
        # print( len(colour_list))
        # for p_index in range(number_of_points):
        #     plt.scatter(x_y_points[p_index][0], x_y_points[p_index][1], s=10, color=colour_list[p_index])#, marker=marker[colour.index(colour_list[p_index])])
            #if (x_y_points[p_index] not in c_list):
                # print x_y_points[p_index]
                #plt.scatter(x_y_points[p_index][0], x_y_points[p_index][1], s=8, color=colour_list[p_index])#, marker=marker[colour.index(colour_list[p_index])])
        for c_index in range(number_of_center):
            plt.scatter(center[0][c_index], center[1][c_index], s=8, color='k',  marker='o', label = "C" + str(c_index + 1)) #marker[c_index]
        fontP = FontProperties()
        fontP.set_size('small')
        # plt.legend([plt], "title", prop=fontP)
        # plt.legend(loc='upper left')
        # plt.legend(loc = 'upper right', bbox_to_anchor = (1, 0.5) )#, ncol = 2,borderaxespad = 0, frameon = False)
        
        # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5),prop=fontP)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop=fontP, frameon = False)
        plt.tight_layout()
        # plt.legend()
        
        # plt.show()
        if os.path.exists(image_save_path + filename[:-4] +"_run" + str(run_index+1) + '_original.png'):
            os.remove(image_save_path + filename[:-4] +"_run" + str(run_index+1) + '_original.png')
        plt.savefig(image_save_path + filename[:-4] +"_run" + str(run_index+1) + '_original.png', dpi=1000)
        plt.close('all')
    
    g.write(filename + " \t " + str(number_of_points) + " \t " + str(number_of_center) + " \t " +  str(sum_dist/number_of_run) + " \t " + str(elapsed_cluster/number_of_run) +" \n")
g.close()