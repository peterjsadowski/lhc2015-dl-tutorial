__authors__ = "Peter Sadowski"
import os
import gc
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
                  "only supported with PyTables")
import numpy as np
from theano import config
from pylearn2.datasets import dense_design_matrix
from dense_design_matrix import DenseDesignMatrixPyTables
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng

class JETSUBSTRUCTURE(DenseDesignMatrixPyTables):
    """
    Only for faster access there is a copy of hdf5 file in PYLEARN2_DATA_PATH
    but it mean to be only readable.  If you wish to modify the data, you
    should pass a local copy to the path argument.

    Parameters
    ----------
    which_set : WRITEME
    path : WRITEME
    center : WRITEME
    scale : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    preprocessor : WRITEME
    """

    def __init__(self, name, which_set='train', dim = (32,32),
                ):
        # TODO: add dim, normalization, etc.
        self.__dict__.update(locals())
        del self.self
    
        # Load preprocessed data (or make it if necessary).
        path = '/extra/pjsadows0/ml/data/physics/jets/'
        filename = path + '/h5/%s/%s_%dx%d.h5' % (name, which_set, dim[0], dim[1])
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        if not os.path.isfile(filename):
            # hdf5 file does not exist. Make it.
            warnings.warn("Over riding existing file: %s" % (filename))
            self.filters = tables.Filters(complib='blosc', complevel=5)
            self.make_data(name, which_set, dim, filename)
        self.h5file = tables.openFile(filename, mode='r')
        data = self.h5file.getNode('/', "Data")
        #if start is not None or stop is not None:
        #    self.h5file, data = self.resize(self.h5file, start, stop)
        view_converter = dense_design_matrix.DefaultViewConverter((dim[0], dim[1], 1), ('b', 0, 1, 'c'))
        
        super(JETSUBSTRUCTURE, self).__init__(X=data.X, y=data.y) #, view_converter=view_converter)
        self.h5file.flush()

    def make_data(self, name, which_set, dim, filename, shuffle=True):
        """
        Make jet image dataset from txt files.
        """
    
        if name == 'debug0' and which_set=='train':
            N = 1000000
            jets0 = readjets('/extra/pjsadows0/ml/data/physics/jets/download2/dijet_1M.txt', stop=N)
            jets1 = readjets('/extra/pjsadows0/ml/data/physics/jets/download2/ww_1M.txt', stop=N)  # class 1
            jets = jets0 + jets1
            center_jets(jets, original=False)
            align_jets(jets, reflect=True) # Reflect so that max is in positive phi. 
            data_x = discretize(jets, dim=dim, xmax=1.8, ymax=1.8, normalize=True, maxoverlap=None) #TODO:x,y max seem to be np.pi/2 in other papers.
            data_x = data_x.reshape((data_x.shape[0], -1)) # Flatten
            data_y = np.vstack((np.zeros((len(jets0), 1), dtype='float32'), np.ones((len(jets1), 1), dtype='float32'))) # Labels
        elif name == 'debug1' and which_set == 'train':
            N = 100
            jets0 = readjets('/extra/pjsadows0/ml/data/physics/jets/download2/dijet_1M.txt', stop=N)
            jets1 = readjets('/extra/pjsadows0/ml/data/physics/jets/download2/ww_1M.txt', stop=N)  # class 1
            jets = jets0 + jets1
            center_jets(jets, original=False)
            align_jets(jets, reflect=True) # Reflect so that max is in positive phi. 
            data_x = discretize(jets, dim=dim, xmax=1.8, ymax=1.8, normalize=True, maxoverlap=None) #TODO:x,y max seem to be np.pi/2 in other papers.
            data_x = data_x.reshape((data_x.shape[0], -1)) # Flatten
            data_y = np.vstack((np.zeros((len(jets0), 1), dtype='float32'), np.ones((len(jets1), 1), dtype='float32'))) # Labels  
        else:
            raise

        assert data_x.shape[0] == data_y.shape[0]
        nexamples = data_x.shape[0]
        ninputs = data_x.shape[1]
        noutputs = data_y.shape[1]
        h5file, node = self.init_hdf5(filename, ([nexamples, ninputs], [nexamples, noutputs]))

        if shuffle:
            rng = make_np_rng(None, 322, which_method="shuffle")  # For consistency between experiments better to make new random stream.
            index = range(data_x.shape[0])
            rng.shuffle(index)
            data_x = data_x[index, :]
            data_y = data_y[index, :]
        
        JETSUBSTRUCTURE.fill_hdf5(h5file, data_x=data_x, data_y=data_y, node=node)
        h5file.close()

def readjets(fn, stop=np.inf):
    ''' Read jet data file from Daniel.'''
    if fn[-3:] == '.gz':
        fid = gzip.open(fn, 'r')
    else:
        fid = open(fn, 'r')
    data = []
    jet = None
    for line in fid.readlines():
        if len(data) >= stop:
            break
        # Read next line.
        line = line.strip().split()
        if line[0] == 'Jet':
            # New jet description. Previous jet is finished.
            if jet:
                data.append(jet)
            jet = {'truth_label': int(line[1]), 'cells':np.zeros((int(line[2]), 3), dtype='float32')}
        elif line[0] == 'cell':
            jet['cells'][int(line[1]), :] = np.array([float(line[2]), float(line[3]), float(line[4])])
        elif line[0][:2] == 'HL':
            jet[line[0]] = [float(x) for x in line[1:]]
    fid.close()
    return data

def plotjets(jets):
    # Scatter plot of jets.
    for jet in jets:
        w = pd.Series(jet['cells'][:,0])
        colors = w.apply(lambda x: (1,0,0, min(x, 1.0))).tolist()
        x = pd.Series(jet['cells'][:,1])
        y = pd.Series(jet['cells'][:,2])
        plt.scatter(x, y, c=colors)
    plt.xlim([-6, 6])
    if mean(y) > 2:
        plt.ylim([0, 2*pi])
    else:
        plt.ylim([-pi, pi])
    xlabel('Angle 1 (Long axis)')
    ylabel('Angle 2')
    plt.show()

def mean_jet(jet):
    # Compute mean activation of jet in both dimensions.
    w = jet['cells'][:,0] / np.sum(jet['cells'][:,0])
    x = jet['cells'][:,1]
    y = jet['cells'][:,2]
    mx = np.dot(w, x) 
    my = np.dot(w, y)
    return mx, my
def principle_axis_jet(jet):
    ''' Calculate principle axis of jet, as defined in Almeida, arxiv.org/pdf/1501.05968.pdf'''
    E, eta, phi = jet['cells'][:,0], jet['cells'][:,1], jet['cells'][:,2]
    R = np.sqrt(eta**2 + phi**2)
    theta = np.arctan2(np.sum(phi * E / R), np.sum(eta * E / R)) # np.arctan2(y,x) computes angle (1,0), (0,0), (x,y) in +-pi/2
    return theta

def center_jets(jets, original=False):
    ''' Translate jet so that energy centroid is at origin and fix boundary problems.'''
    for jet in jets:
        jet['cells'][:,2] -= jet['cells'][np.argmax(jet['cells'][:,0]),2]  # Center on highest-energy point.
        jet['cells'][:,2] = np.mod(jet['cells'][:,2] + np.pi, np.pi* 2) - np.pi # Make sure everything is within 2pi of 0.
        mx, my = mean_jet(jet)
        jet['cells'][:,2] -= my # Need to wrap around too...
        jet['cells'][:,1] -= mx
        if original:
            # Center at (0,pi) instead of (0,0)
            jet['cells'][:,2] += np.pi 
    return

def align_jets(jets, reflect=False):
    '''
    Rotate jet so that principle axis is pointing down as in Almeida, arxiv.org/pdf/1501.05968.pdf.
    reflect = If True, maximal transverse energy always appears on the right side of the image, as in Cogan. http://arxiv.org/pdf/1407.5675v2.pdf.
    '''
    for jet in jets:
        if jet['cells'].shape[0] <= 1:
            continue
        theta = principle_axis_jet(jet)
        eta, phi = jet['cells'][:,1], jet['cells'][:,2]
        neweta = np.cos(theta) * eta + np.sin(theta) * phi
        newphi = - np.sin(theta) * eta + np.cos(theta) * phi
        jet['cells'][:,1], jet['cells'][:,2] = neweta, newphi
        if reflect:
            phi = jet['cells'][:,2]
            E = jet['cells'][:,0]
            if phi[np.argmax(E)] < 0:
                phi *= -1.0 # Reflect over eta axis.
        
def discretize(jets, dim, xmax, ymax, normalize=True, maxoverlap=None):
    X = np.zeros((len(jets), dim[0], dim[1]), dtype='float32')
    eps = 10**-8
    xbins = np.arange(-xmax, xmax + eps, 2*xmax / float(dim[0]), dtype='float32')
    ybins = np.arange(-ymax, ymax + eps, 2*ymax / float(dim[1]), dtype='float32')
    for i, jet in enumerate(jets):
        #img = np.zeros(dim)
        assert min(jet['cells'][:,1]) > xbins[0] and max(jet['cells'][:,1]) < xbins[-1], (i, min(jet['cells'][:,1]), max(jet['cells'][:,1]))
        assert min(jet['cells'][:,2]) > ybins[0] and max(jet['cells'][:,2]) < ybins[-1], (i, min(jet['cells'][:,2]), max(jet['cells'][:,2]))
        img, bx, by = np.histogram2d(jet['cells'][:,1], jet['cells'][:,2], bins=(xbins, ybins), range=None, normed=False, weights=jet['cells'][:,0])
        img2, bx2, by2 = np.histogram2d(jet['cells'][:,1], jet['cells'][:,2], bins=(xbins, ybins), range=None, normed=False)
        if maxoverlap and (np.max(img2) > maxoverlap):
            raise Exception('Overlap of %d events on img %d.' % (np.max(img2), i))
        assert np.all(bx == xbins) and np.all(by == ybins)
        if normalize:
            img = img / np.sum(img.flatten())
        X[i, :, :] = img
    return X
