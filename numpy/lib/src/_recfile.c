/*
 * Defines the Recfile class.  This c class should not be used
 * directly, as very little error checking is performed.  Instead
 * use the numpy.lib.recfile.Recfile object, which inherits from this
 * one and does all the type and error checking.
 */

#include <string.h>
#include <Python.h>
#include <numpy/arrayobject.h> 

struct PyRecfileObject {
    PyObject_HEAD
    // we will keep a reference to file_obj
    PyObject* file_obj;
    FILE* fptr;

    // the delimiter, NULL for binary
    const char* delim;

    // One element for each column of the file, in order.
    // must be contiguous npy_intp arrays
    PyObject* typecodes_obj;
    PyObject* sizes_obj;
    // jost pointers to the data sections
    npy_intp* typecodes;
    npy_intp* sizes;

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
        PyFile_IncUseCount((PyFileObject*)self->file_obj);
#endif
	} else {
        PyErr_SetString(PyExc_IOError, "Input must be an open file object");
        return 0;
	}
    return 1;
}

static npy_intp _get_nrows(PyObject* nrows_obj) {
    // assuming it is an npy_intp array
    npy_intp nrows=0;
    nrows = *(npy_intp* ) PyArray_GETPTR1(nrows_obj, 0);
    return nrows;
}

static int
PyRecfileObject_init(struct PyRecfileObject* self, PyObject *args, PyObject *kwds)
{
    PyObject* nrows_obj=NULL; // if sent, 1 element npy_intp array
    self->file_obj=NULL;
    self->typecodes_obj=NULL;
    self->typecodes=NULL;
    self->sizes_obj=NULL;
    self->sizes=NULL;
    self->nrows=0;
    self->nfields=0;
    if (!PyArg_ParseTuple(args, 
                          (char*)"OOsOO", 
                          &self->file_obj, 
                          &nrows_obj,
                          &self->delim,
                          &self->typecodes_obj,
                          &self->sizes_obj)) {
        return -1;
    }

    if (!_set_fptr(self)) {
        return -1;
    }
    self->nrows = _get_nrows(nrows_obj);
    self->test=7;
    return 0;
}

static PyObject *
PyRecfileObject_repr(struct PyRecfileObject* self) {
    return PyString_FromFormat("test: %d\nnrows: %ld\ndelim: '%s'\n", 
            self->test, self->nrows, self->delim);
}

static PyObject *
PyRecfileObject_test(struct PyRecfileObject* self) {
    return PyInt_FromLong((long)self->test);
}

static void
_cleanup(struct PyRecfileObject* self)
{

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    //fprintf(stderr,"dec use fptr\n");
    // both these introduced in python 2.6
    if (self->file_obj) {
        PyFile_DecUseCount((PyFileObject*)self->file_obj);
    }
    Py_XDECREF(self->file_obj);
#else
    // old way, removed in python 3
    Py_XDECREF(self->file_obj);
#endif

    self->fptr=NULL;
    self->file_obj=NULL;
}


static void
PyRecfileObject_dealloc(struct PyRecfileObject* self)
{
    _cleanup(self);
#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif
}


static PyMethodDef PyRecfileObject_methods[] = {
    {"test",             (PyCFunction)PyRecfileObject_test,             METH_VARARGS,  "test\n\nReturn test value."},
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
