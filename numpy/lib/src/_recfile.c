/*
 * Defines the Recfile class.  This c class should not be used
 * directly, as very little error checking is performed.  Instead
 * use the numpy.lib.recfile.Recfile object, which inherits from this
 * one and does all the type and error checking.
 */

#define NPY_NO_DEPRECATED_API

#include <string.h>
#include <Python.h>
#include "numpy/arrayobject.h" 

struct PyRecfileObject {
    PyObject_HEAD
    // we will keep a reference to file_obj
    PyObject* file_obj;
    FILE* fptr;

    int is_ascii;
    // One element for each column of the file, in order.
    // must be contiguous npy_intp arrays
    //PyObject* typecodes_obj;
    PyObject* sizes_obj;
    PyObject* nel_obj;
    PyObject* scan_formats_obj;
    PyObject* print_formats_obj;

    // jost pointers to the data sections
    npy_intp* sizes;
    npy_intp* nel;

    // rowsize in bytes for binary
    npy_intp rowsize;

    npy_intp nrows; // rows from the input offset of file
    npy_intp nfields;
    int test;
};

/*
 * Check validity of input file object.  We will keep a reference to the file
 * object so it won't get closed on us.  In newer versions of python we can
 * also be thread safe.
 */
static int _set_fptr(struct PyRecfileObject* self)
{
    if (PyFile_Check(self->file_obj)) {
		self->fptr = PyFile_AsFile(self->file_obj);
		if (NULL==self->fptr) {
            PyErr_SetString(PyExc_IOError, "Input file object is invalid");
            return 0;
		}

        Py_XINCREF(self->file_obj);
#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
        //fprintf(stderr,"inc use fptr\n");
        //PyFile_IncUseCount((PyFileObject*)self->file_obj);
#endif
	} else {
        PyErr_SetString(PyExc_IOError, "Input must be an open file object");
        return 0;
	}
    return 1;
}

static npy_intp get_nrows(PyObject* nrows_obj) {
    npy_intp nrows=0;
    if (nrows_obj == Py_None) {
        // this shouldn't happen
        nrows=-1;
    } else {
        // assuming it is an npy_intp array, nonzero len
        // safer since old python didn't support Py_ssize_t
        // in PyArg_ParseTuple
        //nrows = *(npy_intp* ) PyArray_GETPTR1((PyArrayObject*)nrows_obj, 0);
        nrows = *(npy_intp* ) PyArray_DATA((PyArrayObject*)nrows_obj);
    }
    return nrows;
}

static void set_rowsize(struct PyRecfileObject* self) {
    npy_intp i=0;

    self->rowsize=0;
    for (i=0; i<self->nfields; i++) {
        self->rowsize += self->sizes[i];
    }
}

static int
PyRecfileObject_init(struct PyRecfileObject* self, PyObject *args, PyObject *kwds)
{
    PyObject* nrows_obj=NULL; // if sent, 1 element npy_intp array
    self->file_obj=NULL;
    self->is_ascii=0;
    self->scan_formats_obj=NULL;
    self->print_formats_obj=NULL;
    self->sizes_obj=NULL;
    self->sizes=NULL;
    self->rowsize=0;
    self->nel_obj=NULL;
    self->nel=NULL;
    self->nrows=0;
    self->nfields=0;
    if (!PyArg_ParseTuple(args, 
                          (char*)"OiOOOOO", 
                          &self->file_obj, 
                          &self->is_ascii,
                          &self->sizes_obj,
                          &self->nel_obj,
                          &self->scan_formats_obj,
                          &self->print_formats_obj,
                          &nrows_obj)) {
        return -1;
    }

    if (!_set_fptr(self)) {
        return -1;
    }
    self->nrows = get_nrows(nrows_obj);
    self->test=7;
    Py_XINCREF(self->sizes_obj);
    Py_XINCREF(self->nel_obj);
    Py_XINCREF(self->scan_formats_obj);
    Py_XINCREF(self->print_formats_obj);

    self->nfields = PyArray_SIZE((PyArrayObject*)self->sizes_obj);
    self->sizes=(npy_intp*) PyArray_DATA((PyArrayObject*)self->sizes_obj);
    self->nel=(npy_intp*) PyArray_DATA((PyArrayObject*)self->nel_obj);

    set_rowsize(self);
    return 0;
}

static PyObject *
PyRecfileObject_repr(struct PyRecfileObject* self) {
    return PyString_FromFormat("test: %d\nnrows: %ld\n", 
            self->test, self->nrows);
}

static PyObject *
PyRecfileObject_test(struct PyRecfileObject* self) {
    return PyInt_FromLong((long)self->test);
}


int skip_binary_bytes(FILE* fptr, npy_intp nbytes) {
    if (nbytes > 0) {
        return fseeko(fptr, (off_t) nbytes, SEEK_CUR); 
    }
    return 0;
}

static PyObject* 
read_binary_slice(struct PyRecfileObject* self,
                  void* ptr, npy_intp start, npy_intp stop, npy_intp step) {

    npy_intp current_row=0, row=0;
    npy_intp bytes=0;

    if (start > 0) {
        if (0 != skip_binary_bytes(self->fptr, start*self->rowsize) ) {
            // TODO use err string
            PyErr_SetString(PyExc_IOError, "failed to seek");
            return NULL;
        }
    }

    current_row=start;
    row=start;
    while (row < stop) {
        if (row > current_row) {
            bytes = (row-current_row)*self->rowsize;
            if (0 != skip_binary_bytes(self->fptr, bytes)) {
                // TODO use err string
                PyErr_SetString(PyExc_IOError, "failed to seek");
                return NULL;
            }
            current_row=row;
        }

        //fprintf(stderr,"reading row: %ld\n", row);
        if (1 != fread(ptr, (size_t)self->rowsize, 1, self->fptr)) {
            // TODO use err string
            PyErr_Format(PyExc_IOError, "failed to read row %ld", row);
            return NULL;
        }
        //fprintf(stderr,"success\n");

        ptr += self->rowsize;
        current_row += 1;
        row += step;
    }
    Py_RETURN_NONE;

}

/* 
 * read row slice into input array.  As usual, error checking must be done in
 * the python wrapper!  The exception is matching the itemsize of the
 * input array to the rowsize in order to avoid out of bounds errors.
 *
 * we expect slices with step != 1 to be handled in the python wrapper.
 */

static PyObject* 
PyRecfileObject_read_slice(struct PyRecfileObject* self, PyObject* args)
{
    PyObject* array=NULL;
    PyObject* start_obj=NULL;
    PyObject* stop_obj=NULL;
    PyObject* step_obj=NULL;
    npy_intp start=0, stop=0, step=0;
    npy_intp itemsize=0;
    void* ptr=NULL;
    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                          &array,
                          &start_obj,
                          &stop_obj,
                          &step_obj)) {
        return NULL;
    }

    start = *(npy_intp*) PyArray_DATA((PyArrayObject*)start_obj);
    stop = *(npy_intp*) PyArray_DATA((PyArrayObject*)stop_obj);
    step = *(npy_intp*) PyArray_DATA((PyArrayObject*)step_obj);

    itemsize = PyArray_ITEMSIZE((PyArrayObject*) array);
    if (itemsize != self->rowsize) {
        PyErr_Format(PyExc_IOError, 
                "input itemsize (%ld) != rowsize (%ld)",
                itemsize, self->rowsize);
        return NULL;
    }

    ptr = PyArray_DATA((PyArrayObject*) array);
    if (self->is_ascii) {
        fprintf(stderr,"implement ascii slicing\n");
        return NULL;
    } else {
        return read_binary_slice(self, ptr, start, stop, step);
    }
}

static void
cleanup(struct PyRecfileObject* self)
{

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    //fprintf(stderr,"dec use fptr\n");
    // both these introduced in python 2.6
    if (self->file_obj) {
        //PyFile_DecUseCount((PyFileObject*)self->file_obj);
    }
    Py_XDECREF(self->file_obj);
#else
    // old way, removed in python 3
    Py_XDECREF(self->file_obj);
#endif

    self->fptr=NULL;
    self->file_obj=NULL;
    Py_XDECREF(self->sizes_obj);
    Py_XDECREF(self->nel_obj);
    Py_XDECREF(self->scan_formats_obj);
    Py_XDECREF(self->print_formats_obj);

    self->sizes_obj=NULL;
    self->sizes=NULL;
    self->nel_obj=NULL;
    self->nel=NULL;
    self->scan_formats_obj=NULL;
    self->print_formats_obj=NULL;
}


static void
PyRecfileObject_dealloc(struct PyRecfileObject* self)
{

    cleanup(self);

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif
}


// just a function in the _recfile module
// type=1 for scan formats
// type=other for print formats
static PyObject*
PyRecfile_get_formats(PyObject* self, PyObject *args) {
    PyObject* dict=NULL;
    int type=0;

    if (!PyArg_ParseTuple(args, (char*)"i", &type)) {
        //PyErr_SetString(PyExc_IOError, "expected integer");
        return NULL;
    }

    dict = PyDict_New();
    PyDict_SetItem(dict, PyInt_FromLong(NPY_INT8), PyString_FromString(NPY_INT8_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_UINT8), PyString_FromString(NPY_UINT8_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_INT16), PyString_FromString(NPY_INT16_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_UINT16), PyString_FromString(NPY_UINT16_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_INT32), PyString_FromString(NPY_INT32_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_UINT32), PyString_FromString(NPY_UINT32_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_INT64), PyString_FromString(NPY_INT64_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_UINT64), PyString_FromString(NPY_UINT64_FMT));

#ifdef NPY_INT128
    PyDict_SetItem(dict, PyInt_FromLong(NPY_INT128), PyString_FromString(NPY_INT128_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_UINT128), PyString_FromString(NPY_UINT128_FMT));
#endif
#ifdef NPY_INT256
    PyDict_SetItem(dict, PyInt_FromLong(NPY_INT256), PyString_FromString(NPY_INT256_FMT));
    PyDict_SetItem(dict, PyInt_FromLong(NPY_UINT256), PyString_FromString(NPY_UINT256_FMT));
#endif

    if (type == 1) {
        // scan formats
        PyDict_SetItem(dict, PyInt_FromLong(NPY_FLOAT32), PyString_FromString("f"));
        PyDict_SetItem(dict, PyInt_FromLong(NPY_FLOAT64), PyString_FromString("lf"));

#ifdef NPY_FLOAT128
        PyDict_SetItem(dict, PyInt_FromLong(NPY_FLOAT128), PyString_FromString("Lf"));
#endif
    } else {
        // print formats
        PyDict_SetItem(dict, PyInt_FromLong(NPY_FLOAT32), PyString_FromString(".7g"));
        PyDict_SetItem(dict, PyInt_FromLong(NPY_FLOAT64), PyString_FromString(".16g"));

#ifdef NPY_FLOAT128
        // what should this be?
        PyDict_SetItem(dict, PyInt_FromLong(NPY_FLOAT128), PyString_FromString(".16g"));
#endif

        PyDict_SetItem(dict, PyInt_FromLong(NPY_STRING), PyString_FromString("s"));
    }
    return dict;
}


static PyMethodDef PyRecfileObject_methods[] = {
    {"test",             (PyCFunction)PyRecfileObject_test,             METH_VARARGS,  "test\n\nReturn test value."},
    {"read_slice",             (PyCFunction)PyRecfileObject_read_slice,             METH_VARARGS,  "read a slice into input array. Use for step != 1, otherwise can use numpy.fromfile\n"},
    {NULL}  /* Sentinel */
};


static PyTypeObject PyRecfileType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_recfile.Recfile",             /*tp_name*/
    sizeof(struct PyRecfileObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyRecfileObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyRecfileObject_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Recfile Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyRecfileObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyRecfileObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};

static PyMethodDef recfile_methods[] = {
    {"get_formats",      (PyCFunction)PyRecfile_get_formats,METH_VARARGS,"Get some scan formats, which are best done from C"},
    {NULL}  /* Sentinel */
};



#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_recfile",      /* m_name */
        "Defines the Recfile class and some methods",  /* m_doc */
        -1,                  /* m_size */
        recfile_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_recfile(void) 
{
    PyObject* m;


    PyRecfileType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyRecfileType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyRecfileType) < 0) {
        return;
    }
    m = Py_InitModule3("_recfile", recfile_methods, "Define Recfile type and methods.");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyRecfileType);
    PyModule_AddObject(m, "Recfile", (PyObject *)&PyRecfileType);

    import_array();
}
