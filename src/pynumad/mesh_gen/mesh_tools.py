import numpy as np
import plotly.graph_objects as go
from pynumad.mesh_gen.spatial_grid_list2d import *
from pynumad.mesh_gen.spatial_grid_list3d import *
from pynumad.mesh_gen.element_utils import *

def rotate_vector(vec,axis,angle):
    if(angle < 0.0000000001):
        return vec.copy()
    else:
        axAr = np.array(axis)
        mag = np.linalg.norm(axis)
        unitAxis = (1.0/mag)*axAr
        alp1 = np.zeros((3,3),dtype=float)
        alp1[0] = unitAxis
        i1 = 0
        if(abs(unitAxis[1]) < abs(unitAxis[0])):
            i1 = 1
        if(abs(unitAxis[2]) < abs(unitAxis[i1])):
            i1 = 2
        alp1[1,i1] = np.sqrt(1.0 - alp1[0,i1]*alp1[0,i1])
        for i2 in range(0,3):
            if(i2 != i1):
                alp1[1,i2] = -alp1[0,i1]*alp1[0,i2]/alp1[1,i1]
        alp1[2] = cross_prod(alp1[0], alp1[1])
        theta = angle*np.pi/180.0
        cs = np.cos(theta)
        sn = np.sin(theta)
        alp2 = np.array([[1.0,0.0,0.0],
                         [0.0,cs,-sn],
                         [0.0,sn,cs]])
        rV = np.matmul(alp1,vec)
        rV = np.matmul(alp2,rV)
        rV = np.matmul(rV,alp1)
        return rV

def translate_mesh(meshData,tVec):
    tAr = np.array(tVec)
    nLen = len(meshData['nodes'])
    newNds = np.zeros((nLen,3),dtype=float)
    for i, nd in enumerate(meshData['nodes']):
        newNds[i] = nd + tAr
    meshData['nodes'] = newNds
    return meshData

def rotate_mesh(meshData,pt,axis,angle):
    ptAr = np.array(pt)
    nLen = len(meshData['nodes'])
    newNds = np.zeros((nLen,3),dtype=float)
    for i, nd in enumerate(meshData['nodes']):
        tCrd = nd - ptAr
        rCrd = rotate_vector(tCrd,axis,angle)
        newNds[i] = ptAr + rCrd
    meshData['nodes'] = newNds
    return meshData

def get_direction_cosines(xDir,xyDir):
    mag = np.linalg.norm(xDir)
    a1 = (1.0/mag)*xDir
    zDir = cross_prod(xDir,xyDir)
    mag = np.linalg.norm(zDir)
    a3 = (1.0/mag)*zDir
    a2 = cross_prod(a3,a1)
    dirCos = np.array([a1,a2,a3])
    return dirCos

def get_average_node_spacing(nodes, elements):
    """Mean Euclidean distance between every ordered pair of distinct nodes
    within each element. Vectorised over the element list.

    Equivalent to the previous triple Python loop but ~100x faster on
    large meshes (was a profile hotspot at 1.8 s on 25k elements).
    """
    nodes_arr = np.asarray(nodes, dtype=float)
    elements_arr = np.asarray(elements, dtype=int)
    n_elem, k = elements_arr.shape
    if n_elem == 0 or k < 2:
        return 0.0

    valid = elements_arr >= 0
    safe_idx = np.where(valid, elements_arr, 0)
    coords = nodes_arr[safe_idx]  # (n_elem, k, 3)

    total_dist = 0.0
    count = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            mask = valid[:, i] & valid[:, j]
            if not mask.any():
                continue
            diff = coords[:, i, :] - coords[:, j, :]
            d = np.sqrt((diff * diff).sum(axis=1))
            total_dist += d[mask].sum()
            count += int(mask.sum())
    if count == 0:
        return 0.0
    return total_dist / count

def check_all_jacobians(nodes,elements):
    failedEls = set()
    ei = 0
    for el in elements:
        xC = []
        yC = []
        zC = []
        for nd in el:
            if(nd > -1):
                xC.append(nodes[nd,0])
                yC.append(nodes[nd,1])
                zC.append(nodes[nd,2])
        elCrd = np.array([xC,yC,zC])
        nn = len(xC)
        if(nn == 8):
            elType = 'brick8'
        elif(nn == 6):
            elType = 'wedge6'
        else:
            elType = ''
        passed = check_jacobian(elCrd,elType)
        if(not passed):
            failedEls.add(ei)
        ei = ei + 1
    return failedEls


def _check_element_jacobian(nodes, el):
    """Single-element wrapper around check_jacobian — used by the
    untangling pass to test one brick / wedge at a time without
    rebuilding the per-element coordinate matrix in the caller."""
    xC, yC, zC = [], [], []
    for nd in el:
        if nd > -1:
            xC.append(nodes[nd, 0])
            yC.append(nodes[nd, 1])
            zC.append(nodes[nd, 2])
    nn = len(xC)
    if nn == 8:
        etype = 'brick8'
    elif nn == 6:
        etype = 'wedge6'
    else:
        return True  # unknown — assume OK
    return check_jacobian(np.array([xC, yC, zC]), etype)


def untangle_solid_mesh(nodes, elements, max_iter=30,
                        factor_schedule=(0.5, 0.3, 0.15, 0.05),
                        verbose=False):
    """Post-extrusion untangling pass for swept solid meshes.

    Standard industry stage after smoothing + adaptive layer thickness:
    iteratively repair the residual bad-Jacobian bricks/wedges by
    pulling each element's TOP-face corners toward their corresponding
    BOTTOM-face corners (pure through-thickness shrink), with a
    neighbour-preservation guard so previously-good elements never
    silently invert. Implements the targeted "node-pull" stage that
    Mesquite-style optimizers do as a final pass.

    Convention from ``Mesh3D.createSweptMesh``:
      hex   — corners 0..3 are the bottom face (shell base layer),
              4..7 the top face (extruded layer). Top corner k+4 sits
              above bottom corner k.
      wedge — bottom 0..2, top 3..5. Top corner k+3 above bottom k.

    Algorithm:
      For each iteration:
        1. Identify bad-Jacobian elements via check_all_jacobians.
        2. For each, try each shrink factor in ``factor_schedule``:
             move top corners to bot + f * (top - bot).
             accept the first factor where (a) the element becomes
             good AND (b) no neighbour that was good before turns bad.
        3. Stop when zero bad elements or no move accepted.

    Returns
    -------
    new_nodes : np.ndarray
        Repaired nodes array (copy; input unchanged).
    n_remaining : int
        Bad elements still failing after ``max_iter`` rounds.
    n_iterations : int
        Number of iterations actually run.

    Performance
    -----------
    Bounded by ``max_iter * len(bad)``; for BAR0 at elementSize=0.5
    after smoothing + clamp this is ~2 * 8 = 16 single-element
    Jacobian checks plus per-node neighbour lookups (~200 ops total).
    """
    nodes = nodes.copy()
    n = len(nodes)
    node_to_els = [[] for _ in range(n)]
    for ei in range(len(elements)):
        for nid in elements[ei]:
            if nid >= 0:
                node_to_els[nid].append(ei)

    for it in range(max_iter):
        bad = check_all_jacobians(nodes, elements)
        if not bad:
            if verbose:
                print(f'untangle: iter {it}: 0 bad — converged')
            return nodes, 0, it
        good_initial = set(range(len(elements))) - set(bad)
        moved = 0
        for ei in list(bad):
            el = elements[ei]
            is_wedge = (el[6] == -1)
            top_idx = [3, 4, 5] if is_wedge else [4, 5, 6, 7]
            bot_idx = [0, 1, 2] if is_wedge else [0, 1, 2, 3]
            for factor in factor_schedule:
                snapshot = {el[k]: nodes[el[k]].copy() for k in top_idx}
                for ti, bi in zip(top_idx, bot_idx):
                    nodes[el[ti]] = (nodes[el[bi]] +
                                     factor * (nodes[el[ti]] - nodes[el[bi]]))
                ok = _check_element_jacobian(nodes, el)
                if ok:
                    for ti in top_idx:
                        for other_ei in node_to_els[el[ti]]:
                            if (other_ei in good_initial and
                                    not _check_element_jacobian(nodes, elements[other_ei])):
                                ok = False
                                break
                        if not ok:
                            break
                if ok:
                    moved += 1
                    break
                for nid, pos in snapshot.items():
                    nodes[nid] = pos
        if verbose:
            print(f'untangle: iter {it}: bad={len(bad)} moved={moved}')
        if moved == 0:
            break
    n_remaining = len(check_all_jacobians(nodes, elements))
    return nodes, n_remaining, it + 1


def get_element_volumes(meshData):
    elVols = dict()
    elMats = dict()
    nodes = meshData['nodes']
    elements = meshData['elements']
    elSets = meshData['sets']['element']
    if(len(elements[0]) > 4):
        for i, sec in enumerate(meshData['sections']):
            for ei in elSets[i]['labels']:
                elCrd = get_el_coord(elements[ei],nodes)
                if(elements[ei][6] == -1):
                    elType = 'wedge6'
                else:
                    elType = 'brick8'
                eVol = get_volume(elCrd,elType)
                stei = str(ei+1)
                elVols[stei] = eVol
                elMats[stei] = sec['material']
    else:
        for i, sec in enumerate(meshData['sections']):
            for ei in elSets[i]['labels']:
                elCrd = get_el_coord(elements[ei],nodes)
                if(elements[ei][3] == -1):
                    elType = 'shell3'
                else:
                    elType = 'shell4'
                eVol = get_volume(elCrd,elType)
                layVol = list()
                layMat = list()
                for lay in sec['layup']:
                    layVol.append(eVol*lay[1])
                    layMat.append(lay[0])
                stei = str(ei+1)
                elVols[stei] = layVol
                elMats[stei] = layMat
    
    output = dict()
    output['elVols'] = elVols
    output['elMats'] = elMats
    return output

def get_mesh_spatial_list(nodes,xSpacing=0,ySpacing=0,zSpacing=0):
    totNds = len(nodes)
    spaceDim = len(nodes[0])

    maxX = np.amax(nodes[:,0])
    minX = np.amin(nodes[:,0])
    maxY = np.amax(nodes[:,1])
    minY = np.amin(nodes[:,1])
    nto1_2 = np.power(totNds,0.5)
    nto1_3 = np.power(totNds,0.333333333333)
    if(spaceDim == 3):
        maxZ = np.amax(nodes[:,2])
        minZ = np.amin(nodes[:,2])
        dimVec = np.array([(maxX-minX),(maxY-minY),(maxZ-minZ)])
        meshDim = np.linalg.norm(dimVec)
        maxX = maxX + 0.01*meshDim
        minX = minX - 0.01*meshDim
        maxY = maxY + 0.01*meshDim
        minY = minY - 0.01*meshDim
        maxZ = maxZ + 0.01*meshDim
        minZ = minZ - 0.01*meshDim
        if(xSpacing == 0):
            xS = 0.5*(maxX - minX)/nto1_3
        else:
            xS = xSpacing
        if(ySpacing == 0):
            yS = 0.5*(maxY - minY)/nto1_3
        else:
            yS = ySpacing
        if(zSpacing == 0):
            zS = 0.5*(maxZ - minZ)/nto1_3
        else:
            zS = zSpacing
        meshGL = spatial_grid_list3d(minX,maxX,minY,maxY,minZ,maxZ,xS,yS,zS)
        #tol = 1.0e-6*meshDim/nto1_3
    else:
        dimVec = np.array([(maxX-minX),(maxY-minY)])
        meshDim = np.linalg.norm(dimVec)
        maxX = maxX + 0.01*meshDim
        minX = minX - 0.01*meshDim
        maxY = maxY + 0.01*meshDim
        minY = minY - 0.01*meshDim
        if(xSpacing == 0):
            xS = 0.5*(maxX - minX)/nto1_2
        else:
            xS = xSpacing
        if(ySpacing == 0):
            yS = 0.5*(maxY - minY)/nto1_2
        else:
            yS = ySpacing
        meshGL = spatial_grid_list2d(minX,maxX,minY,maxY,xS,yS)
        #tol = 1.0e-6*meshDim/nto1_2
    return meshGL

## - Convert list of mesh objects into a single merged mesh, returning sets representing the elements/nodes from the original meshes
def mergeDuplicateNodes(meshData, tolerance=None):
    """Merge nodes within ``tolerance`` of each other into one.

    Vectorised implementation using scipy.cKDTree to find all pairs within
    the tolerance ball in O((N + P) log N) time, where P is the number of
    matching pairs. Replaces the previous get_mesh_spatial_list-based
    routine that was a profile hotspot at ~2.9 s on 25k elements.

    Preserves the original semantics: the surviving representative for any
    cluster of coincident nodes is the one with the smallest original
    index; later duplicates are remapped to the survivor.
    """
    allNds = np.asarray(meshData["nodes"], dtype=float)
    allEls = np.asarray(meshData["elements"], dtype=int)
    totNds = allNds.shape[0]
    if totNds == 0:
        return meshData

    if tolerance is None:
        avgSp = get_average_node_spacing(meshData["nodes"], meshData["elements"])
        tol = 1.0e-4 * avgSp
    else:
        tol = tolerance
    if tol <= 0.0:
        meshData["nodes"] = allNds
        meshData["elements"] = allEls
        return meshData

    # Local import keeps scipy as a soft dep at this site (numpy is hard).
    from scipy.spatial import cKDTree

    tree = cKDTree(allNds)
    # query_pairs returns sorted (i, j) tuples with i < j, all within `tol`.
    pairs = tree.query_pairs(r=tol)

    # ndElim[i] = j means node i is eliminated in favor of node j (j < i).
    # By processing pairs in (i,j) order with j<i and taking the smallest
    # surviving representative, we match the original loop's "if n2i > n1i"
    # collapse semantics.
    ndElim = -np.ones(totNds, dtype=int)
    for i, j in pairs:
        # We want the smaller index to be the survivor.
        lo, hi = (i, j) if i < j else (j, i)
        # Follow the chain — if `lo` is already eliminated, point to its root.
        root = lo
        while ndElim[root] != -1:
            root = ndElim[root]
        if root < hi and ndElim[hi] == -1:
            ndElim[hi] = root

    # Build the survivor remap.
    ndNewInd = -np.ones(totNds, dtype=int)
    survivor_mask = ndElim == -1
    ndNewInd[survivor_mask] = np.arange(int(survivor_mask.sum()))
    nodesFinal = allNds[survivor_mask]

    # Remap element node ids in a single vectorised pass (preserving -1
    # sentinels). For eliminated nodes, follow the chain to the survivor.
    if pairs:
        # Resolve chains so ndElim points directly to survivors.
        for n in np.flatnonzero(~survivor_mask):
            chain = n
            while ndElim[chain] != -1:
                chain = ndElim[chain]
            ndElim[n] = chain

    # Build a per-node "new index" table: survivors -> their new id,
    # eliminated -> survivor's new id.
    direct_new = ndNewInd.copy()
    elim_mask = ~survivor_mask
    direct_new[elim_mask] = ndNewInd[ndElim[elim_mask]]

    # Apply to elements, preserving -1 sentinels.
    valid_mask = allEls >= 0
    new_els = np.where(valid_mask, direct_new[np.where(valid_mask, allEls, 0)], -1)

    meshData["nodes"] = nodesFinal
    meshData["elements"] = new_els

    return meshData
    
def merge_meshes(mData1,mData2,tolerance=None):
    mergedData = dict()
    nds1 = mData1['nodes']
    nLen1 = len(nds1)
    nds2 = mData2['nodes']
    nLen2 = len(nds2)
    totNds = nLen1 + nLen2
    mrgNds = np.zeros((totNds,3),dtype=float)
    mrgNds[0:nLen1] = nds1
    mrgNds[nLen1:totNds] = nds2
    els1 = mData1['elements']
    eLen1 = len(els1)
    els2 = mData2['elements']
    eLen2 = len(els2)
    totEls = eLen1 + eLen2
    eCols = len(els1[0])
    mrgEls = -1*np.ones((totEls,eCols),dtype=int)
    mrgEls[0:eLen1] = els1
    for i, el in enumerate(els2,start=eLen1):
        addVec = np.zeros(eCols,dtype=int)
        for j, nd in enumerate(el):
            if(nd > -1):
                addVec[j] = nLen1
        mrgEls[i] = el + addVec
    mergedData['nodes'] = mrgNds
    mergedData['elements'] = mrgEls
    mergedData['sets'] = dict()
    mergedData['sets']['node'] = list()
    mergedData['sets']['element'] = list()
    try:
        mergedData['sets']['node'].extend(mData1['sets']['node'])
    except:
        pass
    try:
        mergedData['sets']['element'].extend(mData1['sets']['element'])
    except:
        pass
    try:
        for ns in mData2['sets']['node']:
            newSet = dict()
            newSet['name'] = ns['name']
            labs = list()
            for nd in ns['labels']:
                labs.append(nd+nLen1)
            newSet['labels'] = labs
            mergedData['sets']['node'].append(newSet)
    except:
        pass
    try:
        for es in mData2['sets']['element']:
            newSet = dict()
            newSet['name'] = es['name']
            labs = list()
            for el in es['labels']:
                labs.append(el+eLen1)
            newSet['labels'] = labs
            mergedData['sets']['element'].append(newSet)
    except:
        pass
    return mergeDuplicateNodes(mergedData,tolerance)

def add_node_set(meshData,newSet):
    try:
        meshData['sets']['node'].append(newSet)
    except:
        nSets = list()
        nSets.append(newSet)
        try:
            meshData['sets']['node'] = nSets
        except:
            sets = dict()
            sets['node'] = nSets
            meshData['sets'] = sets
    return meshData

def add_element_set(meshData,newSet):
    try:
        meshData['sets']['element'].append(newSet)
    except:
        elSets = list()
        elSets.append(newSet)
        try:
            meshData['sets']['element'] = elSets
        except:
            sets = dict()
            sets['element'] = elSets
            meshData['sets'] = sets
    return meshData

def get_matching_node_sets(meshData):
    elements = meshData['elements']
    elSets = meshData['sets']['element']
    nodeSets = list()
    for es in elSets:
        ns = set()
        for ei in es['labels']:
            for elnd in elements[ei]:
                if(elnd > -1):
                    ns.add(elnd)
        newSet = dict()
        newSet['name'] = es['name']
        newSet['labels'] = list(ns)
        nodeSets.append(newSet)
    try:
        meshData['sets']['node'].extend(nodeSets)
    except:
        meshData['sets']['node'] = nodeSets
        
    return meshData

def get_extruded_sets(meshData,numLayers):
    numEls = len(meshData['elements'])
    numNds = len(meshData['nodes'])
    extSets = dict()
    try:
        elSets = meshData['sets']['element']
        extES = list()
        for es in elSets:
            labels = list()
            for lay in range(0,numLayers):
                for ei in es['labels']:
                    newLab = ei + numEls*lay
                    labels.append(newLab)
            newSet = dict()
            newSet['name'] = es['name']
            newSet['labels'] = labels
            extES.append(newSet)
        extSets['element'] = extES
    except:
        pass
    
    try:
        ndSets = meshData['sets']['node']
        extNS = list()
        for ns in ndSets:
            labels = list()
            for lay in range(0,(numLayers + 1)):
                for ni in ns['labels']:
                    newLab = ni + numNds*lay
                    labels.append(newLab)
            newSet = dict()
            newSet['name'] = ns['name']
            newSet['labels'] = labels
            extNS.append(newSet)
        extSets['node'] = extNS
    except:
        pass        
        
    return extSets

def get_element_set_union(meshData,setList,newSetName):
    un = set()
    for es in meshData['sets']['element']:
        if(es['name'] in setList):
            thisSet = set(es['labels'])
            un = un.union(thisSet)
    newSet = dict()
    newSet['name'] = newSetName
    newSet['labels'] = list(un)
    meshData['sets']['element'].append(newSet)
    return meshData

def make3D(meshData):
    numNodes = len(meshData["nodes"])
    nodes3D = np.zeros((numNodes, 3))
    nodes3D[:, 0:2] = meshData["nodes"]
    dataOut = dict()
    dataOut["nodes"] = nodes3D
    dataOut["elements"] = meshData["elements"]
    return dataOut

def get_surface_nodes(meshData,elSet,newSetName,normDir,normTol=5.0):
    nds = meshData['nodes']
    els = meshData['elements']
    mag = np.linalg.norm(normDir)
    unitNorm = (1.0/mag)*normDir
    cosTol = np.cos(normTol*np.pi/180.0)
    faceDic = dict()
    for es in meshData['sets']['element']:
        if(es['name'] == elSet):
            for ei in es['labels']:
                fcStr, globFc = get_sorted_face_strings(els[ei])
                for fi, fk in enumerate(fcStr):
                    try:
                        curr = faceDic[fk]
                        faceDic[fk] = None
                    except:
                        faceDic[fk] = globFc[fi]
    surfSet = set()
    for fk in faceDic:
        glob = faceDic[fk]
        if(glob is not None):
            gLen = len(glob)
            if(gLen == 3):
                v1 = nds[glob[1]] - nds[glob[0]]
                v2 = nds[glob[2]] - nds[glob[1]]
            elif(gLen == 4):
                v1 = nds[glob[2]] - nds[glob[0]]
                v2 = nds[glob[3]] - nds[glob[1]]
            cp = cross_prod(v1,v2)
            mag = np.linalg.norm(cp)
            fcNrm = (1.0/mag)*cp
            dp = np.dot(fcNrm,unitNorm)
            if(dp >= cosTol):
                for nd in glob:
                    surfSet.add(nd)
    newSet = dict()
    newSet['name'] = newSetName
    newSet['labels'] = list(surfSet)
    return add_node_set(meshData,newSet)

def tie_2_meshes_constraints(tiedMesh,tiedSetName,tgtMesh,tgtSetName,maxDist):
    tiedNds = tiedMesh['nodes']
    tgtNds = tgtMesh['nodes']
    tgtEls = tgtMesh['elements']
    for ns in tiedMesh['sets']['node']:
        if(ns['name'] == tiedSetName):
            tiedSet = ns['labels']
    for es in tgtMesh['sets']['element']:
        if(es['name'] == tgtSetName):
            tgtSet = es['labels']
    radius = get_average_node_spacing(tgtNds,tgtEls)
    elGL = get_mesh_spatial_list(tgtNds,radius,radius,radius)
    if(radius < maxDist):
        radius = maxDist
    for ei in tgtSet:
        el = tgtEls[ei]
        fstNd = tgtNds[el[0]]
        elGL.addEntry(ei,fstNd)
    solidStr = 'tet4 wedge6 brick8'
    constraints = list()
    for ni in tiedSet:
        nd = tiedNds[ni]
        nearEls = elGL.findInRadius(nd,radius)
        minDist = 1.0e+100
        minPO = dict()
        minEi = -1
        for ei in nearEls:
            if(len(tgtEls[ei]) <= 4):
                if(tgtEls[ei,3] == -1):
                    elType = 'shell3'
                else:
                    elType = 'shell4'
            elif(len(tgtEls[ei]) <= 8):
                if(tgtEls[ei,4] == -1):
                    elType = 'tet4'
                elif(tgtEls[ei,6] == -1):
                    elType = 'wedge6'
                else:
                    elType = 'brick8'
            else:
                pstr = 'Warning: encountered unsupported element type in tie2MeshesConstraints'
                print(pstr)
            xC = []
            yC = []
            zC = []
            for en in tgtEls[ei]:
                if(en > -1):
                    xC.append(tgtNds[en,0])
                    yC.append(tgtNds[en,1])
                    zC.append(tgtNds[en,2])
            elCrd = np.array([xC,yC,zC])
            pO = get_proj_dist(elCrd,elType,nd)
            if(elType in solidStr):
                if(pO['distance'] > 0.0):
                    solidPO = get_solid_surf_proj(elCrd,elType,nd)
                    if(solidPO['distance'] < minDist):
                        minDist = solidPO['distance']
                        minPO = solidPO
                        minEi = ei
                else:
                    minDist = 0.0
                    minPO = pO
                    minEi = ei
            else:
                if(pO['distance'] < minDist):
                    minDist = pO['distance']
                    minPO = pO
                    minEi = ei
        if(minDist < maxDist):
            newConst = dict()
            terms = list()
            newTerm = dict()
            newTerm['nodeSet'] = 'tiedMesh'
            newTerm['node'] = ni
            newTerm['coef'] = -1.0
            terms.append(newTerm)
            nVec = minPO['nVec']
            nVi = 0
            for en in tgtEls[minEi]:
                if(en > -1):
                    newTerm = dict()
                    newTerm['nodeSet'] = 'targetMesh'
                    newTerm['node'] = en
                    newTerm['coef'] = nVec[nVi]
                    terms.append(newTerm)
                    nVi = nVi + 1
            newConst['terms'] = terms
            newConst['rhs'] = 0.0
            constraints.append(newConst)
    return constraints

def tie_2_sets_constraints(mesh,tiedSetName,tgtSetName,maxDist):
    try:
        elements = mesh['elements']
        nodes = mesh['nodes']
        elSets = mesh['sets']['element']
        ndSets = mesh['sets']['node']
        for es in elSets:
            if(es['name'] == tgtSetName):
                tgtSet = es['labels']
        for ns in ndSets:
            if(ns['name'] == tiedSetName):
                tiedSet = ns['labels']
            if(ns['name'] == tgtSetName):
                tgtNdSet = ns['labels']
        fstEl = tgtSet[0]
        fstNd = tgtSet[0]
        fstTgtNd = tgtNdSet[0]
    except:
        raise Exception('There was a problem accessing the mesh data in tie2SetsConstraints().  Check the set names and make sure nodes, elements and sets exist in the input mesh')

    tgtNdCrd = []
    for ni in tgtNdSet:
        tgtNdCrd.append(nodes[ni])
    tgtNdCrd = np.array(tgtNdCrd)
    
    radius = get_average_node_spacing(nodes, elements)
    elGL = get_mesh_spatial_list(tgtNdCrd,radius,radius,radius)
    if(radius < maxDist):
        radius = maxDist
    for ei in tgtSet:
        fstNd = tgtNdCrd[elements[ei,0]]
        elGL.addEntry(ei,fstNd)
        ei = ei + 1    
    
    solidStr = 'tet4 wedge6 brick8'
    constraints = list()
    for ni in tiedSet:
        nd = nodes[ni]
        nearEls = elGL.findInRadius(nd,radius)
        minDist = 1.0e+100
        minPO = dict()
        minEi = -1
        for ei in nearEls:
            if(len(elements[ei]) <= 4):
                if(elements[ei,3] == -1):
                    elType = 'shell3'
                else:
                    elType = 'shell4'
            elif(len(elements[ei]) <= 8):
                if(elements[ei,4] == -1):
                    elType = 'tet4'
                elif(elements[ei,6] == -1):
                    elType = 'wedge6'
                else:
                    elType = 'brick8'
            else:
                pstr = 'Warning: encountered unsupported element type in tie2SetsConstraints'
                print(pstr)
            xC = []
            yC = []
            zC = []
            for en in elements[ei]:
                if(en > -1):
                    xC.append(nodes[en,0])
                    yC.append(nodes[en,1])
                    zC.append(nodes[en,2])
            elCrd = np.array([xC,yC,zC])
            pO = get_proj_dist(elCrd,elType,nd)
            if(elType in solidStr):
                if(pO['distance'] > 0.0):
                    solidPO = get_solid_surf_proj(elCrd,elType,nd)
                    if(solidPO['distance'] < minDist):
                        minDist = solidPO['distance']
                        minPO = solidPO
                        minEi = ei
                else:
                    minDist = 0.0
                    minPO = pO
                    minEi = ei
            else:
                if(pO['distance'] < minDist):
                    minDist = pO['distance']
                    minPO = pO
                    minEi = ei
        if(minDist < maxDist and (ni not in elements[minEi])):
            newConst = dict()
            terms = list()
            newTerm = dict()
            newTerm['nodeSet'] = tiedSetName
            newTerm['node'] = ni
            newTerm['coef'] = -1.0
            terms.append(newTerm)
            nVec = minPO['nVec']
            nVi = 0
            for en in elements[minEi]:
                if(en > -1):
                    newTerm = dict()
                    newTerm['nodeSet'] = tgtSetName
                    newTerm['node'] = en
                    newTerm['coef'] = nVec[nVi]
                    terms.append(newTerm)
                    nVi = nVi + 1
            newConst['terms'] = terms
            newConst['rhs'] = 0.0
            constraints.append(newConst)
    
    return constraints

def plotNodes(meshData):
    xLst = meshData['nodes'][:,0]
    yLst = meshData['nodes'][:,1]
    try:
        zLst = meshData['nodes'][:,2]
    except:
        zLst = np.zeros(len(xLst),dtype=float)
    
    fig = go.Figure(data=[go.Scatter3d(x=xLst,y=yLst,z=zLst,mode='markers')])
    
    fig.show()

def plotShellMesh(meshData):
    xLst = meshData["nodes"][:, 0]
    yLst = meshData["nodes"][:, 1]
    try:
        zLst = meshData["nodes"][:, 2]
    except:
        zLst = np.zeros(len(xLst))
    value = list()
    v1 = list()
    v2 = list()
    v3 = list()
    i = 0
    for el in meshData["elements"]:
        v1.append(el[0])
        v2.append(el[1])
        v3.append(el[2])
        value.append(np.sin(i))
        if el[3] != -1:
            v1.append(el[0])
            v2.append(el[2])
            v3.append(el[3])
            value.append(np.sin(i))
        i = i + 1
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=xLst,
                y=yLst,
                z=zLst,
                colorbar_title="",
                colorscale=[[0.0, "white"], [0.5, "gray"], [1.0, "black"]],
                intensity=value,
                intensitymode="cell",
                i=v1,
                j=v2,
                k=v3,
                name="",
                showscale=True,
            )
        ]
    )

    fig.show()

def plotSolidMesh(meshData):
    xLst = meshData["nodes"][:, 0]
    yLst = meshData["nodes"][:, 1]
    zLst = meshData["nodes"][:, 2]
    value = list()
    v1 = list()
    v2 = list()
    v3 = list()
    i = 0
    for el in meshData["elements"]:
        si = np.sin(i)
        if el[4] == -1:
            v1.append(el[0])
            v2.append(el[1])
            v3.append(el[2])
            value.append(si)
            v1.append(el[0])
            v2.append(el[1])
            v3.append(el[3])
            value.append(si)
            v1.append(el[0])
            v2.append(el[2])
            v3.append(el[3])
            value.append(si)
            v1.append(el[1])
            v2.append(el[2])
            v3.append(el[3])
            value.append(si)
        elif el[6] == -1:
            v1.append(el[0])
            v2.append(el[1])
            v3.append(el[2])
            value.append(si)
            v1.append(el[3])
            v2.append(el[4])
            v3.append(el[5])
            value.append(si)
            v1.append(el[0])
            v2.append(el[1])
            v3.append(el[3])
            value.append(si)
            v1.append(el[1])
            v2.append(el[3])
            v3.append(el[4])
            value.append(si)

            v1.append(el[0])
            v2.append(el[2])
            v3.append(el[3])
            value.append(si)
            v1.append(el[2])
            v2.append(el[3])
            v3.append(el[5])
            value.append(si)
            v1.append(el[1])
            v2.append(el[2])
            v3.append(el[4])
            value.append(si)
            v1.append(el[2])
            v2.append(el[4])
            v3.append(el[5])
            value.append(si)
        else:
            v1.append(el[0])
            v2.append(el[3])
            v3.append(el[4])
            value.append(si)
            v1.append(el[3])
            v2.append(el[4])
            v3.append(el[7])
            value.append(si)
            v1.append(el[1])
            v2.append(el[2])
            v3.append(el[5])
            value.append(si)
            v1.append(el[2])
            v2.append(el[5])
            v3.append(el[6])
            value.append(si)

            v1.append(el[0])
            v2.append(el[1])
            v3.append(el[4])
            value.append(si)
            v1.append(el[1])
            v2.append(el[4])
            v3.append(el[5])
            value.append(si)
            v1.append(el[2])
            v2.append(el[3])
            v3.append(el[6])
            value.append(si)
            v1.append(el[3])
            v2.append(el[6])
            v3.append(el[7])
            value.append(si)

            v1.append(el[0])
            v2.append(el[1])
            v3.append(el[2])
            value.append(si)
            v1.append(el[0])
            v2.append(el[2])
            v3.append(el[3])
            value.append(si)
            v1.append(el[4])
            v2.append(el[5])
            v3.append(el[6])
            value.append(si)
            v1.append(el[4])
            v2.append(el[6])
            v3.append(el[7])
            value.append(si)
        i = i + 1
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=xLst,
                y=yLst,
                z=zLst,
                colorbar_title="",
                colorscale=[[0.0, "white"], [0.5, "gray"], [1.0, "black"]],
                intensity=value,
                intensitymode="cell",
                i=v1,
                j=v2,
                k=v3,
                name="",
                showscale=True,
            )
        ]
    )

    fig.show()


## -Create node/element set within a spatial range or radius
