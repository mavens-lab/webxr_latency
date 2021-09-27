'''
 * Author:    Carter Slocum
 * Created:   08.01.2021
 * University of California, Riverside
 * (c) Creative Commons Attribution
'''

import os
import json
import sys
import math
import numpy as np
import quaternion
import copy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

###################################################Constants###################################################
ANG2RAD = 3.14159265358979323846/180.0 #ratio of degrees to radians
EPS = 0.0000001 #epsilon for floating point math

#both of these dictionarys should match the gltf 2.0 specification for type and component type size. mostly here for debug purposes
COMPONENTTYPESIZE = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4
}
TYPESIZE = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16
}

###################################################Parsing Funcitons###################################################

#calculate camera params that are derived from basic params
def DeriveCamera(cam):
    cam["tan"] = math.tan(ANG2RAD * cam["fov"] * 0.5)
    cam["nearheight"] = cam["near"] * cam["tan"]
    cam["nearwidth"] = cam["nearheight"] * cam["ratio"]
    cam["farheight"] = cam["far"] * cam["tan"]
    cam["farwidth"] = cam["farheight"] * cam["ratio"]
    return cam

#parse GLTF file into JSON object
def ParseGLTF(GLTF_Name):
    #parses .gltf file and returnes the bounding boxes and associated images (textures,normals ect) usend in that AABB
    gltf_file = open(GLTF_Name)
    gltf = json.load(gltf_file)
    gltf_file.close()

    return gltf

#from the JSON obtain data of interest
def GetAssets(gltf, cam):
    Images = [] #Materials[x] = [baseColorTextureIndex, metallicRoughnessTextureIndex, normalTextureIndex]
    
    for material in gltf['materials']:
        textures = [] #-1 is not possible in gltf format and in this case is a sentinel value for no texture
        if 'pbrMetallicRoughness' in material:
            if 'baseColorTexture' in material['pbrMetallicRoughness']:
                textures.append((gltf['images'][gltf['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]['source']]['uri'], material['pbrMetallicRoughness']['baseColorTexture']['index']))
            if 'metallicRoughnessTexture' in material['pbrMetallicRoughness']:
                textures.append((gltf['images'][gltf['textures'][material['pbrMetallicRoughness']['metallicRoughnessTexture']['index']]['source']]['uri'], material['pbrMetallicRoughness']['metallicRoughnessTexture']['index']))
        if 'normalTexture' in material:
            textures.append((gltf['images'][gltf['textures'][material['normalTexture']['index']]['source']]['uri'], material['normalTexture']['index']))
        if 'extensions' in material:
            if 'diffuseTexture' in material["extensions"]["KHR_materials_pbrSpecularGlossiness"]:
                textures.append((gltf['images'][gltf['textures'][material["extensions"]["KHR_materials_pbrSpecularGlossiness"]["diffuseTexture"]["index"]]['source']]['uri'], material["extensions"]["KHR_materials_pbrSpecularGlossiness"]["diffuseTexture"]["index"]))
        Images.append(textures)

    AABBs = [] #each element is [xmin , ymin, zmin, xmax, ymax, zmax]
    Materials = [] #int corresponds to index of associated Image

    for mesh in gltf['meshes']:
        for primitive in mesh['primitives']:
            BB = gltf['accessors'][primitive['attributes']['POSITION']]['min']
            BB.extend(gltf['accessors'][primitive['attributes']['POSITION']]['max'])
            AABBs.append(BB)
            Materials.append(primitive['material'])

    Assets = []
    for mat in Materials:
        Assets.append(Images[mat])

    for node in gltf['nodes']:
        if 'name' in node and node['name'] == "Camera":
            Camera[0] = node['translation']
            Camera[1] = node['rotation'] 

    tex = 0
    if 'textures' in gltf:
        tex = len(gltf['textures'])
    return Assets, tex,  AABBs, cam

################################################### View Geometry Code ###################################################

#calculate the center of the AABB
def CenterAABB(aabb):
    x = aabb[0] + ((aabb[3] - aabb[0])/2)
    y = aabb[1] + ((aabb[4] - aabb[1])/2)
    z = aabb[2] + ((aabb[5] - aabb[2])/2)
    return [x, y, z]

# get distance of center of aabb from camera position
def ScoreDistance(aabbs, cam):
    distances = []
    camp = cam["center"]
    for aabb in aabbs:
        abc = CenterAABB(aabb)
        x = camp[0] - abc[0]
        y = camp[1] - abc[1]
        z = camp[2] - abc[2]
        d = math.sqrt(x*x + y*y + z*z)
        distances.append(d)
    return distances

def Mag3(Vec3):
    return math.sqrt(Vec3[0]*Vec3[0] + Vec3[1]*Vec3[1] + Vec3[2]*Vec3[2])

#3d Normalized vector
def Normalize3(Vec3):
    mag = Mag3(Vec3)
    return [Vec3[0]/mag, Vec3[1]/mag, Vec3[2]/mag]

#3d dot product
def Dot3(Vec3a, Vec3b):
    dot = 0.0
    for i in range(len(Vec3a)):
        dot += Vec3a[i]*Vec3b[i]
    return dot

#3d crossproduct
def Cross3(Vec3a, Vec3b):
    return [Vec3a[1]*Vec3b[2] - Vec3a[2]*Vec3b[1], Vec3a[2]*Vec3b[0] -Vec3a[0]*Vec3b[2], Vec3a[0]*Vec3b[1] - Vec3a[1]*Vec3b[0]]

#get quaternion from axis angle presentation
def QuatFromAxisAngle(axis,angle):
    halfangle = angle * 0.5
    s = math.sin(halfangle)
    return [axis[0] * s, axis[1] * s, axis[2] *s, math.cos(halfangle)]

#scale a 3d Vector by a scalar
def ScaleVec3(vec, scale):
    return np.array([vec[0]*scale,vec[1]*scale,vec[2]*scale])

#get all 6 planes [near,far,top,bot,left,right] that compose the cameras view frustum
def ViewFrustum(cam):
    Z = cam["center"] - quaternion.rotate_vectors(cam["orientation"],np.array([0.0,0.0,1.0]))
    Z = Normalize3(Z)
    X = Normalize3(Cross3(np.array([0.0,1.0,0.0]), Z))
    Y = Normalize3(Cross3(Z,X))
    nearcenter = cam["center"] + ScaleVec3(Z,cam["near"])
    farcenter = cam["center"] + ScaleVec3(Z,cam["far"])

    nearp = [np.array([-z for z in Z]), nearcenter]
    farp = [np.array(Z), farcenter]

    #plane comprised of [Normal,Point]
    topp = [Cross3(Normalize3(nearcenter + ScaleVec3(Y,cam["nearheight"])) - cam["center"], X), nearcenter + ScaleVec3(Y,cam["nearheight"])]
    botp = [Cross3(X,Normalize3(nearcenter - ScaleVec3(Y,cam["nearheight"])) - cam["center"]), nearcenter - ScaleVec3(Y,cam["nearheight"])]
    leftp = [Cross3(Normalize3(nearcenter - ScaleVec3(X,cam["nearwidth"])) - cam["center"], Y), nearcenter - ScaleVec3(X,cam["nearwidth"])]
    rightp = [Cross3(Y,Normalize3(nearcenter + ScaleVec3(X,cam["nearwidth"])) - cam["center"]), nearcenter + ScaleVec3(X,cam["nearwidth"])]

    return [nearp, farp, topp, botp, leftp, rightp]

#get the signed minimum distance from a point to a plane
def DistanceToPlane(plane, point):
    return  0.0 - plane[0][0]*(point[0] - plane[1][0]) - plane[0][1]*(point[1] - plane[1][1]) - plane[0][2]*(point[2] - plane[1][2])

#return True if AABB intersects viewfrustrum (or is entirely contained within) False otherwise
def AABBFrustumCheck(aabb, frustrum):
    #make list of all 8 vertecies on the AABB
    points = [np.array([aabb[0],aabb[1],aabb[2]]),
    np.array([aabb[3],aabb[1],aabb[2]]),
    np.array([aabb[0],aabb[4],aabb[2]]),
    np.array([aabb[0],aabb[1],aabb[5]]),
    np.array([aabb[3],aabb[4],aabb[2]]),
    np.array([aabb[0],aabb[4],aabb[5]]),
    np.array([aabb[3],aabb[1],aabb[5]]),
    np.array([aabb[3],aabb[4],aabb[5]])]

    #following algorithm exploits the fact that if there exists no plane in a frustrum where all vertecies lie outside of then the box MUST intersect or be contained within the view Frustrum
    outside = []

    planeOutside = []

    for plane in frustum:
        outsides = []
        for point in points:
            distance = DistanceToPlane(plane, point)
            if distance < -EPS: #point is outside the plane
                outsides.append(True)
            else: #point is inside the plane
                outsides.append(False)
        if False in outsides: # see above comment on properties of box intersections
            planeOutside.append(False)
        else:
            planeOutside.append(True)
    if True in planeOutside:
        return False
    return True

#calculate Quaternion roatation to look from vector VecFrom to vector VecTo
def LookAtQuaternion(VecFrom, VecTo):
    forward = Normalize3([VecTo[0] - VecFrom[0],VecTo[1] - VecFrom[1],VecTo[2] - VecFrom[2]])
    dot = Dot3([0.0,0.0,-1.0],forward)

    if (abs(dot - (-1.0)) < EPS): #directly behind
        return [0.0, 1.0, 0.0, 0.0]
    if (abs(dot - (1.0)) < EPS): #directly in front
        return [0.0,0.0,0.0,1.0]

    angle = math.acos(dot)
    axis = Cross3([0.0,0.0,-1.0], forward)
    axis = Normalize3(axis)
    return QuatFromAxisAngle(axis,angle)

#calculate size of rotation (radians) needed for camera to view the desired point
def ScoreOrientation(aabbs, cam):
    rotations = []
    for aabb in aabbs:
        aabbV = CenterAABB(aabb)
        lookatV = Normalize3([aabbV[0] - cam["center"][0], aabbV[1] - cam["center"][1], aabbV[2] - cam["center"][2]])
        lookingV = Normalize3(quaternion.rotate_vectors(cam["orientation"],np.array([0.0,0.0,-1.0])))
        rotations.append(math.acos(Dot3(lookatV,lookingV)/(Mag3(lookingV)*Mag3(lookatV))))
    return rotations

#combine scores into priority heuristic
def VisabilityScores(vals, angles):
    scores = []
    for i in range(len(vals)):
        scores.append(angles[i] + 2.0*vals[i])
    return scores

#graph GLTF AABBs and camera with scores
def DebugAABB(aabb, camera, scores, frustum):
    plt.title("Camera To Board Vectors")
    ax = plt.axes(projection='3d')
    abc = [CenterAABB(bb) for bb in aabb]
    ax.set_xlabel('X', fontsize=35, labelpad = 35)
    ax.set_ylabel('Z', fontsize=35, labelpad = 35)
    ax.set_zlabel('Y', fontsize=35, labelpad = 35)
    xs = np.array([c[0] for c in abc])
    ys = np.array([c[1] for c in abc])
    zs = np.array([c[2] for c in abc])


    cs = []
    smax = max(scores)
    smin = min(scores)
    srange = smax - smin

    for s in scores:
        slerp = (s - smin)/srange
        cs.append(np.array([0.0,0.0,slerp]))

    cs = np.array(cs)
    #color should be bright blue for high priority and nearly black for low priority

    print('--------------------------')
    print(scores)
    print('--------------------------')

    ax.set_box_aspect((np.ptp(xs),np.ptp(zs),np.ptp(ys)))

    ax.scatter3D(xs,zs,ys, c=cs)

    camc = camera["center"]
    x0q, y0q, z0q = np.meshgrid(camc[0],camc[1],camc[2])

    '''
    x0q = np.array([camera[0][0]])
    y0q = np.array([camera[0][1]])
    z0q = np.array([camera[0][2]])
    '''
    Z = camera["center"] - quaternion.rotate_vectors(camera["orientation"],np.array([0.0,0.0,-1.0]))
    Z = Normalize3(Z)


    #x1q = Z[0]
    #y1q = Z[1]
    #z1q = Z[2]

    x2q = frustum[0][0][0]
    y2q = frustum[0][0][1]
    z2q = frustum[0][0][2] 

    #ax.quiver(x0q, z0q, y0q, x1q, z1q, y1q, length = 1000.0, normalize=True, color='red')
    ax.quiver(x0q, z0q, y0q, x2q, z2q, y2q, length = 1000.0, normalize=True, color='red')
    for plane in frustum[1:]:
        ax.quiver(x0q, z0q, y0q, plane[0][0], plane[0][2], plane[0][1], length = 1000.0, normalize=True, color='green')

    plt.tick_params(labelsize='20')

    plt.show()

#check indecies of images and textures
def checkIndex(img, tex): #debug function
    print(len(img))
    print(len(tex))
    for i in range(len(img)):
        if -1 in img[i]:
            print(i)
    for i in range(len(tex)):
        if -1 in tex[i]:
            print(i)

###################################################Image Prioritization Code###################################################

#put sorted images into JSON object
def PrioritizeImages(imgs,tex_count):
    #neeed to show which image is now what
    pimg = []
    imgset = set()
    ind_fix = np.zeros(tex_count, dtype=int)
    img_count = 0
    for imgtri in range(len(imgs)):
        for img in range(len(imgs[imgtri])):
            if imgs[imgtri][img] != -1 and imgs[imgtri][img][0] not in imgset:
                image = {
                    'uri': imgs[imgtri][img][0]
                }
                imgset.add(imgs[imgtri][img][0])
                pimg.append(image)
                ind_fix[img_count] = imgs[imgtri][img][1]
                img_count += 1
    return pimg, ind_fix

#dump debug info to text file
def OutToText(outFile, scores, index, imgs, aabbs):
    #index:score:images:aabb\n  (maybe add byte range for the bin later)
    f = open(outFile, "w")
    for i in range(len(index)):
        f.write(str(index[i]) + ":")
        f.write(str(scores[i]) + ":")
        f.write(str(imgs[i]) + ":")
        f.write(str(aabbs[i]) + "\n")
    f.close()



#fix the texture index used for the materials after the images have been sorted and replace the images list with the sorted one
def FixGLTF(gltf, pimg, ind_fix, prim_fix):

    '''
    #fix textures (not recomended)
    for i in range(len(gltf['textures'])):
        gltf['textures'][i]['source'] = int(ind_fix[gltf['textures'][i]['source']])
    '''

    #fix materials texture indecies
    for m in range(len(gltf['materials'])):
        if 'pbrMetallicRoughness' in gltf['materials'][m]:
            if 'baseColorTexture' in gltf['materials'][m]['pbrMetallicRoughness']:
                gltf['materials'][m]['pbrMetallicRoughness']['baseColorTexture']['index'] = int(np.where(ind_fix == gltf['materials'][m]['pbrMetallicRoughness']['baseColorTexture']['index'])[0][0])
            if 'metallicRoughnessTexture' in gltf['materials'][m]['pbrMetallicRoughness']:
                gltf['materials'][m]['pbrMetallicRoughness']['metallicRoughnessTexture']['index'] = int(np.where(ind_fix == gltf['materials'][m]['pbrMetallicRoughness']['metallicRoughnessTexture']['index'])[0][0])
        if 'normalTexture' in gltf['materials'][m]:
            gltf['materials'][m]['normalTexture']['index'] = int(np.where(ind_fix == gltf['materials'][m]['normalTexture']['index'])[0][0])
        if 'extensions' in gltf['materials'][m]:
            if 'diffuseTexture' in gltf['materials'][m]["extensions"]["KHR_materials_pbrSpecularGlossiness"]:
                gltf['materials'][m]["extensions"]["KHR_materials_pbrSpecularGlossiness"]["diffuseTexture"]["index"] = int(np.where(ind_fix == gltf['materials'][m]["extensions"]["KHR_materials_pbrSpecularGlossiness"]["diffuseTexture"]["index"])[0][0])

    #fix order of primatives within the mesh
    prims = []
    for mesh in gltf["meshes"]:
        for prim in mesh.keys():
            if prim == 'primitives':
                for ob in mesh[prim]:
                    prims.append(copy.deepcopy(ob))
    meshes = []
    primitive = {
        "primitives" : []
    }

    meshes.append(primitive)
    for i in range(len(prim_fix)):
        meshes[0]["primitives"].append(prims[prim_fix[i]])

    gltf["meshes"] = meshes

    #fix image order
    gltf['images'] = pimg

    return gltf

#output prioritized GLTF
def OutNewGLTF(outFile, gltf):
    with open(outFile, 'w') as file:
        json.dump(gltf, file)

###################################################Bin Partition Code###################################################

#partitions the binary file into the ranges in the list of tuples
def PartitionBin(name, byte_ranges):
    files = 0
    binary = 0
    with open(str(name) + ".bin", mode='rb') as binf:
        binary = binf.read()

    while files < len(byte_ranges):
        chunks = []
        for br in byte_ranges[files]:
            rg = br[1] - br[0]
            rg = rg + (16 - (rg % 16)) % 16
            chunks.append(binary[br[0]:br[0] + rg])
        with open(str(name) + str(files) + ".bin", mode='wb') as chunk_file:
            for chunk in chunks:
                chunk_file.write(chunk)
        files += 1

#helper function mostly for calculating total .bin partition sizes (not used)
def SumRanges(ranges):
    summation = 0
    for r in ranges:
        summation += r[1] - r[0]
    return summation

#helper function mostly for calculating total .bin partition sizes (used)
def SumAlignedRanges(ranges):
    summation = 0
    for r in ranges:
        rg = r[1] - r[0]
        rg = rg + (16 - (rg % 16)) % 16
        summation += rg
    return summation

###################################################GLTF Partition Code###################################################

#copy extensions
def FixExtensions(ingltf, outgltf):
    if 'extensionsRequired' in ingltf:
        outgltf["extensionsRequired"] = copy.deepcopy(ingltf["extensionsRequired"])
    if 'extensionsUsed' in ingltf:
        outgltf["extensionsUsed"] = copy.deepcopy(ingltf["extensionsUsed"])

#copy Asset GLTF class
def FixAsset(ingltf, outgltf): 
    if 'asset' in ingltf:
        outgltf["asset"] = copy.deepcopy(ingltf["asset"])

#make new node (must create heirarchy in the javascript node additions to the scene))
def FixNodes(ingltf, outgltf):
    if 'nodes' in ingltf:
        outgltf["nodes"] = [copy.deepcopy(ingltf["nodes"][0])]
        if "children" in outgltf["nodes"][0]:
            del outgltf["nodes"][0]["children"]
        outgltf["nodes"][0]["mesh"] = 0

#set buffer to .bin partition
def FixBuffers(outgltf, name, buffer_number, byte_ranges):
    outgltf["buffers"] = [{
        "byteLength": SumAlignedRanges(byte_ranges),
        "uri": name + str(buffer_number) + ".bin"
    }]

#set views into buffer SET 1 VIEW PER ACCESSOR
def FixBufferViews(ingltf, outgltf, indi, ranges):
    outgltf["bufferViews"] = []
    offset = 0
    for i in range(len(ranges)):
        r = ranges[i][1] - ranges[i][0]
        view = {
            "buffer": 0,
            "byteLength": r,
            "byteOffset": offset
        }
        outgltf["bufferViews"].append(copy.deepcopy(view))
        offset += r
        offset = offset + (16 - (offset % 16)) % 16 #necessary to maintain 16 byte alignment

#set accessors into bufferviews (must work with FixBufferViews)
def FixAccessors(ingltf, outgltf, indi):
    in_accessors = copy.deepcopy(ingltf["accessors"])
    outgltf["accessors"] = []
    for i in range(len(indi)):
        outgltf["accessors"].append(in_accessors[indi[i]])
        outgltf["accessors"][i]["bufferView"] = i
        outgltf["accessors"][i]["byteOffset"] = 0

#copy over samplers
def FixSamplers(ingltf, outgltf):
    if "samplers" in ingltf:
        outgltf["samplers"] = copy.deepcopy(ingltf["samplers"])

#fix materials to use only those for this primative in the correct order
def FixImages(ingltf, outgltf, buffer_number):
    if 'textures' not in ingltf:
        return
    mesh = ingltf["meshes"][0]["primitives"][buffer_number]
    mat = copy.deepcopy(ingltf["materials"][mesh["material"]])
    in_tex = copy.deepcopy(ingltf["textures"])
    in_img = copy.deepcopy(ingltf["images"])
    tex = []
    if "normalTexture" in mat:
        tex.append(mat["normalTexture"]["index"])
    if "pbrMetallicRoughness" in mat:
        if "baseColorTexture" in mat["pbrMetallicRoughness"]:
            tex.append(mat["pbrMetallicRoughness"]["baseColorTexture"]["index"])
        if "metallicRoughnessTexture" in  mat["pbrMetallicRoughness"]:
            tex.append(mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"])
    if 'extensions' in mat:
        if 'KHR_materials_pbrSpecularGlossiness' in mat['extensions']:
            if 'diffuseTexture' in mat['extensions']["KHR_materials_pbrSpecularGlossiness"]:
                tex.append(mat['extensions']["KHR_materials_pbrSpecularGlossiness"]["diffuseTexture"]["index"])
    images = []
    for t in tex:
        images.append(in_tex[t]["source"])
    if len(images) > 0:
        outgltf["images"] = []
        for i in images:
            outgltf["images"].append(in_img[i])

#fix txtures to only use those for the material.
def FixTextures(ingltf, outgltf, buffer_number):
    if 'textures' not in ingltf:
        return
    mesh = ingltf["meshes"][0]["primitives"][buffer_number]
    mat = copy.deepcopy(ingltf["materials"][mesh["material"]])
    in_tex = copy.deepcopy(ingltf["textures"])

    tex = []
    if "normalTexture" in mat:
        tex.append(mat["normalTexture"]["index"])
    if "pbrMetallicRoughness" in mat:
        if "baseColorTexture" in mat["pbrMetallicRoughness"]:
            tex.append(mat["pbrMetallicRoughness"]["baseColorTexture"]["index"])
        if "metallicRoughnessTexture" in  mat["pbrMetallicRoughness"]:
            tex.append(mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"])
    if 'extensions' in mat:
        if 'KHR_materials_pbrSpecularGlossiness' in mat['extensions']:
            if 'diffuseTexture' in mat['extensions']["KHR_materials_pbrSpecularGlossiness"]:
                tex.append(mat['extensions']["KHR_materials_pbrSpecularGlossiness"]["diffuseTexture"]["index"])
    if len(tex) > 0:
        outgltf["textures"] = []
        for t in range(len(tex)):
            outgltf["textures"].append(copy.deepcopy(in_tex[tex[t]]))
            outgltf["textures"][t]["source"] = t
    

#only one material per gltf, correct the textures
def FixMaterials(ingltf, outgltf, buffer_number):
    mesh = ingltf["meshes"][0]["primitives"][buffer_number]
    mat = copy.deepcopy(ingltf["materials"][mesh["material"]])
    outgltf["materials"] = [mat]
    index = 0
    if "normalTexture" in outgltf["materials"][0]:
        outgltf["materials"][0]["normalTexture"]["index"] = index
        index += 1
    if "pbrMetallicRoughness" in outgltf["materials"][0]:
        if "baseColorTexture" in outgltf["materials"][0]["pbrMetallicRoughness"]:
            outgltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] = index
            index += 1
        if "metallicRoughnessTexture" in outgltf["materials"][0]["pbrMetallicRoughness"]:
            outgltf["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"] = index
            index += 1
    if 'extensions' in outgltf["materials"][0]:
        if 'KHR_materials_pbrSpecularGlossiness' in outgltf["materials"][0]["extensions"]:
            if 'diffuseTexture' in outgltf["materials"][0]["extensions"]["KHR_materials_pbrSpecularGlossiness"]:
                outgltf["materials"][0]["extensions"]["KHR_materials_pbrSpecularGlossiness"]["diffuseTexture"]["index"] = index
                index += 1

#ideally one discrete mesh per gltf. (origional is list of meshes which is list of primitives)
def FixMeshes(ingltf, outgltf, buffer_number):
    outgltf["meshes"] = []
    outgltf["meshes"].append({})
    outgltf["meshes"][0]["primitives"] = [copy.deepcopy(ingltf["meshes"][0]["primitives"][buffer_number])]
    index = 0
    outgltf["meshes"][0]["primitives"][0]["material"] = 0
    if "indices" in outgltf["meshes"][0]["primitives"][0]:
        outgltf["meshes"][0]["primitives"][0]["indices"] = index
        index += 1
    for key in outgltf["meshes"][0]["primitives"][0]["attributes"].keys():
        outgltf["meshes"][0]["primitives"][0]["attributes"][key] = index
        index += 1

#copy over
def FixScene(ingltf, outgltf):
    outgltf["scene"] = copy.deepcopy(ingltf["scene"])

#copy over
def FixScenes(ingltf, outgltf):
    outgltf["scenes"] = [
        {
            "name": "Scene",
            "nodes": [
                0
            ]
        }
    ]

#orgonize order of partitioning 
def PartitionGLTF(gltf, name, ind, byte_ranges):
    for b in range(len(byte_ranges)):
        outgltf = {}
        FixExtensions(gltf, outgltf)
        FixAsset(gltf, outgltf)
        FixNodes(gltf, outgltf)
        FixBuffers(outgltf, name, b, byte_ranges[b])
        FixBufferViews(gltf, outgltf, ind[b], byte_ranges[b])
        FixAccessors(gltf, outgltf, ind[b])
        FixSamplers(gltf, outgltf)
        FixImages(gltf, outgltf, b)
        FixTextures(gltf, outgltf, b)
        FixMaterials(gltf, outgltf, b)
        FixMeshes(gltf, outgltf, b)
        FixScene(gltf, outgltf)
        FixScenes(gltf, outgltf)
        OutNewGLTF(name + str(b) + ".gltf",outgltf)


#splits the binary and splits the gltf to reflect
def SplitPrims(fileName, gltf):
    primatives = []
    for mesh in gltf["meshes"]:
        for primative in mesh:
            primatives.append(mesh[primative])

    #get indices of buffer views for each primative
    indices = []
    for prim in primatives[0]:
        buffer_indicies = [prim["indices"]]
        for key in prim["attributes"].keys():
            buffer_indicies.append(prim["attributes"][key])
        indices.append(buffer_indicies)

    #get the byte ranges for the bins
    primatives = []
    partitions = []
    for ind in indices:
        byte_ranges = []
        for i in range(len(ind)):
            start = 0
            bufferview = gltf["accessors"][ind[i]]["bufferView"]
            if "byteOffset" in gltf["accessors"][ind[i]]:
                start = gltf["accessors"][ind[i]]["byteOffset"]
            start += gltf["bufferViews"][bufferview]["byteOffset"]
            comptype = COMPONENTTYPESIZE[gltf["accessors"][ind[i]]["componentType"]]
            counttype = gltf["accessors"][ind[i]]["count"]
            type = TYPESIZE[gltf["accessors"][ind[i]]["type"]]
            end = start + (comptype * counttype * type)
            byte_ranges.append((start, end))
        partitions.append(byte_ranges)


    #partition the bin
    PartitionBin(fileName[:-5],partitions)

    PartitionGLTF(gltf, fileName[:-5], indices, partitions)

#check if byte offsets are 16Byte Alligned (Debug / GLTF Validation)
def CheckByteAlignment(gltf):
    for buff in gltf["bufferViews"]:
        if buff["byteOffset"] % 16 > 0:
            return False
    return True

################################################### Main ###################################################

if __name__ == "__main__":
    GLTF_Name = sys.argv[1]
    gltf = ParseGLTF(GLTF_Name)

    '''
    valid = CheckByteAlignment(gltf)
    if not valid:
        print("Invalid GLTF")
        exit()
    '''

    #edit this to match initial view as needed (defaults to Sponza).
    Camera = {
    "center": np.array([0.0,0.0,0.0]),
    "orientation": np.quaternion(0.7071,0.0,-0.7072,0.0), #example looking down the +x axis
    #"orientation": np.quaternion(0.7071,-0.7072,0.0,0.0), #example looking down the -y axis
    #"orientation": np.quaternion(1.0,0.0,0.0,0.0), #example looking down the -z axis
    #"orientation": np.quaternion(0.7071,0.0,0.7072,0.0), #example looking down the -x axis
    #"orientation": np.quaternion(0.7071,0.7072,0.0,0.0), #example looking down the -y axis
    #"orientation": np.quaternion(0.0,0.0,1.0,0.0), #example looking down the +z axis
    "ratio": 16.0/9.0,
    "fov": 90.0,
    "near": 0.1,
    "far": 10000.0,
    }

    img, tex_count, aabb, Camera =  GetAssets(copy.deepcopy(gltf), Camera)

    DCam = DeriveCamera(Camera)
    frustum = ViewFrustum(DCam)

    angles = ScoreOrientation(aabb,DCam)
    visable = []
    for box in aabb:
        visable.append(AABBFrustumCheck(box,frustum))

    vals = []
    for s in visable:
        if s:
            vals.append(0.0)
        else:
            vals.append(1.0)
    scores = VisabilityScores(vals,angles)

    index = list(range(0,len(scores)))
    scores_s, index_s, img_s, aabb_s = map(list, zip(*sorted(zip(scores,index,img,aabb))))
 

    DebugAABB(aabb_s, DCam, scores_s, frustum)

    pimg, tex_fix = PrioritizeImages(img_s, tex_count)
    gltf = FixGLTF(gltf, pimg, tex_fix, index_s)
    #At this point the gltf is sorted and ready to go but it and the .bin still need to be split up.
    #OutNewGLTF(OUT_NAME, gltf) #only uncomment this for VINS-Image

    #partition into many gltfs and bins
    SplitPrims(GLTF_Name, gltf)