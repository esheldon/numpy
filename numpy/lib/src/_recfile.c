/*
 * Defines the Recfile class.  This c class should not be used directly, as
 * very little error checking is performed.  Instead use the
 * numpy.recfile.Recfile object, which inherits from this one and does all the
 * type and error checking.
 *
 * todo: deal with file object in python3: You can no longer get a FILE*
 * associated with an open file!  All the ascii number reading code relies on
 * the power of fscanf
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

    char* delim;
    int delim_is_space;


    npy_intp nrows; // rows from the input offset of file
    npy_intp ncols;


    // One element for each column of the file, in order.
    // must be contiguous npy_intp arrays
    //PyObject* typecodes_obj;
    PyObject* typenums_obj;
    PyObject* elsize_obj;
    PyObject* nel_obj;
    PyObject* offset_obj;
    PyObject* scan_formats_obj;
    PyObject* print_formats_obj;
    PyObject* converters_obj;

    // jost pointers to the data sections
    npy_intp* typenums;
    npy_intp* elsize;
    npy_intp* nel;
    npy_intp* offset;
    npy_intp* converters;


    // rowsize in bytes for binary
    npy_intp rowsize;


    // reading text
    //
    // strings can be surrounded by quotes.  Must be a one-char string
    // or empty ""  Quoted strings can be variable length.
    char* quote_char;
    int has_quoted_strings;

    // char used to indicate comment
    char* comment_char;

    // strings can be variable length but not quoted; in this case we only keep
    // as many as fit in the fixed width numpy array buffer.
    int has_var_strings;

    // buffer when skipping text fields
    npy_intp buffsize;
    char* buffer;


    // writing text
    // replace ascii string characters beyond null with spaces
    int padnull;
    // don't print chars beyind null, will result in not fixed width cols
    int ignore_null;

    // not yet implemented
    // bracketed arrays for postgres
    int bracket_arrays;

};

/*
 * Check validity of input file object.  We will keep a reference to the file
 * object so it won't get closed on us.
 */
static int _set_fptr(struct PyRecfileObject* self)
{
    int status=1;
    if (PyFile_Check(self->file_obj)) {
		self->fptr = PyFile_AsFile(self->file_obj);
		if (NULL==self->fptr) {
            PyErr_SetString(PyExc_IOError, "Input file object is invalid");
            status = 0;
		} else {
            Py_XINCREF(self->file_obj);
        }
	} else {
        PyErr_SetString(PyExc_IOError, "Input must be an open file object");
        status=0;
	}
    return status;
}



/*
 * calc row size and set buffer for skipping string cols
 */

static int set_rowsize_and_buffer(struct PyRecfileObject* self) {
    npy_intp i=0;
    npy_intp elsize=0, totsize=0;

    self->rowsize=0;
    for (i=0; i<self->ncols; i++) {
        elsize = self->elsize[i];
        totsize = elsize*self->nel[i];
        self->rowsize += totsize;

        if (self->typenums[i] == NPY_STRING) {
            if (elsize > self->buffsize) {
                // some padding
                self->buffsize = elsize + 2;
            }
        }
    }

    self->buffer = (char*) malloc(self->buffsize);
    if (self->buffer == NULL) {
        PyErr_Format(PyExc_IOError, "failed to allocate buffer[%ld]", 
                     self->buffsize);
        return 0;
    }
    return 1;
}

static void 
set_defaults(struct PyRecfileObject* self)
{
    self->file_obj=NULL;

    self->delim=NULL;
    self->delim_is_space=0;
    self->is_ascii=0;

    self->quote_char=NULL;
    self->comment_char=NULL;
    self->has_quoted_strings=0;
    self->has_var_strings=0;

    self->scan_formats_obj=NULL;
    self->print_formats_obj=NULL;
    self->typenums_obj=NULL;
    self->elsize_obj=NULL;
    self->elsize=NULL;
    self->rowsize=0;
    self->nel_obj=NULL;
    self->offset_obj=NULL;
    self->converters_obj=NULL;
    self->offset=NULL;
    self->nel=NULL;
    self->nrows=0;
    self->ncols=0;
    self->bracket_arrays=0;
    self->padnull=0;
    self->ignore_null=0;

    self->buffsize=256; // will expand for large strings

}
static int
PyRecfileObject_init(struct PyRecfileObject* self, PyObject *args, PyObject *kwds)
{
    set_defaults(self);
    if (!PyArg_ParseTuple(args, 
                          (char*)"OsiOOOOOOiissi|O", 
                          &self->file_obj, 
                          &self->delim, 
                          &self->is_ascii,
                          &self->typenums_obj,
                          &self->elsize_obj,
                          &self->nel_obj,
                          &self->offset_obj,
                          &self->scan_formats_obj,
                          &self->print_formats_obj,
                          &self->padnull,
                          &self->ignore_null, 
                          &self->quote_char,
                          &self->comment_char,
                          &self->has_var_strings,
                          &self->converters_obj)) {
        return -1;
    }

    if (!_set_fptr(self)) {
        return -1;
    }
    if (self->is_ascii && strlen(self->delim) != 1) {
        PyErr_Format(PyExc_IOError, 
                "delim for ascii should be 1 character, got '%s'", 
                self->delim);
        return -1;
    }
    if (self->is_ascii) {
        if (strncmp(" ",self->delim,1)==0) {
            self->delim_is_space=1;
        }
        if (NULL != self->quote_char) {
            int n=0;
            n=strlen(self->quote_char);
            if (n > 1) {
                PyErr_Format(PyExc_IOError, 
                        "quote char for ascii should be 1 character, got '%s'", 
                        self->quote_char);
                return -1;
            }
            if (n == 1) {
                self->has_quoted_strings=1;
            }
        }
    }

    self->ncols = PyArray_SIZE((PyArrayObject*)self->elsize_obj);
    self->typenums=(npy_intp*) PyArray_DATA((PyArrayObject*)self->typenums_obj);
    self->elsize=(npy_intp*) PyArray_DATA((PyArrayObject*)self->elsize_obj);
    self->nel=(npy_intp*) PyArray_DATA((PyArrayObject*)self->nel_obj);
    self->offset=(npy_intp*) PyArray_DATA((PyArrayObject*)self->offset_obj);
    self->converters=(npy_intp*) PyArray_DATA((PyArrayObject*)self->converters_obj);

    if (!set_rowsize_and_buffer(self)) {
        return -1;
    }

    Py_XINCREF(self->elsize_obj);
    Py_XINCREF(self->nel_obj);
    Py_XINCREF(self->offset_obj);
    Py_XINCREF(self->scan_formats_obj);
    Py_XINCREF(self->print_formats_obj);

    return 0;
}

static PyObject *
PyRecfileObject_repr(struct PyRecfileObject* self) {
    return PyString_FromFormat(
            "Recfile\n\tdelim: '%s'\n\tnrows: %ld\n\tbuffsize: %ld\n", 
            self->delim, self->nrows, self->buffsize);
}

/*
 * use a npy_intp object here since safer for old pythons
 */
static PyObject *
PyRecfileObject_set_nrows(struct PyRecfileObject* self, PyObject* args) {
    PyObject* nrows_obj;
    if (!PyArg_ParseTuple(args, (char*)"O", &nrows_obj)) {
        return NULL;
    }
    
    self->nrows = *(npy_intp* ) PyArray_DATA((PyArrayObject*)nrows_obj);
    Py_RETURN_NONE;
}



static int
write_string_element(struct PyRecfileObject* self, npy_intp col, char* ptr) {
    int status=1;
    npy_intp i=0;
    int res=0;
    char c;

    if (!self->ignore_null && !self->padnull) {
        // we write the string exactly as it is, null and all
        if (1 != fwrite(ptr, self->elsize[col], 1, self->fptr)) {
            status=0;
        }
    } else {

        for (i=0; i<self->elsize[col]; i++) {
            c = ptr[0];

            if (c == '\0') {
                if (self->ignore_null) {
                    // we assume the user cares about nothing beyond the null
                    // this will break out of writing this the rest of this col.
                    // and result in not-fixed width cols
                    break;
                }
                if (self->padnull) {
                    c=' ';
                }
            }
            res = fputc( (int) c, self->fptr);
            if (res == EOF) {
                status=0;
                break;
            }
            ptr++;
        }
    }
    return status;
}

static int
write_number_element(struct PyRecfileObject* self, npy_intp col, char* ptr) {
    int status=1;
    char *fmt=NULL;
    npy_intp typenum=0;
	int res=0;

    typenum = self->typenums[col];
    fmt = PyArray_GETPTR1((PyArrayObject*)self->print_formats_obj, col);
	switch (typenum) {
		case NPY_INT8:
			res= fprintf( self->fptr, fmt, *(npy_int8* )ptr ); 	
			break;
		case NPY_UINT8:
			res= fprintf( self->fptr, fmt, *(npy_uint8* )ptr ); 	
			break;

		case NPY_INT16:
			res= fprintf( self->fptr, fmt, *(npy_int16* )ptr ); 	
			break;
		case NPY_UINT16:
			res= fprintf( self->fptr, fmt, *(npy_uint16* )ptr ); 	
			break;

		case NPY_INT32:
			res= fprintf( self->fptr, fmt, *(npy_int32* )ptr ); 	
			break;
		case NPY_UINT32:
			res= fprintf( self->fptr, fmt, *(npy_uint32* )ptr ); 	
			break;

		case NPY_INT64:
			res= fprintf( self->fptr, fmt, *(npy_int64* )ptr ); 	
			break;
		case NPY_UINT64:
			res= fprintf( self->fptr, fmt, *(npy_uint64* )ptr ); 	
			break;

#ifdef NPY_INT128
		case NPY_INT128:
			res= fprintf( self->fptr, fmt, *(npy_int128* )ptr ); 	
			break;
		case NPY_UINT128:
			res= fprintf( self->fptr, fmt, *(npy_uint128* )ptr ); 	
			break;
#endif
#ifdef NPY_INT256
		case NPY_INT256:
			res= fprintf( self->fptr, fmt, *(npy_int256* )ptr ); 	
			break;
		case NPY_UINT256:
			res= fprintf( self->fptr, fmt, *(npy_uint256* )ptr ); 	
			break;
#endif

		case NPY_FLOAT32:
			res= fprintf( self->fptr, fmt, *(npy_float32* )ptr ); 	
			break;
		case NPY_FLOAT64:
			res= fprintf( self->fptr, fmt, *(npy_float64* )ptr ); 	
			break;
#ifdef NPY_FLOAT128
		case NPY_FLOAT128:
			res= fprintf( self->fptr, fmt,*(npy_float128* )ptr ); 	
			break;
#endif

		default:
            fprintf(stderr,"Unsupported typenum: %ld\n", typenum);
            res=-1;
	}

	if (res < 0) {
		status=0;
	}

    return status;
}


static int
write_ascii_col_element(struct PyRecfileObject* self, npy_intp col, char* ptr) {
    int status=1;
    if (NPY_STRING == self->typenums[col]) {
        status=write_string_element(self,col,ptr);
    } else {
        status=write_number_element(self,col,ptr);
    }
    return status;
}

static int
write_ascii_col_bracketed(struct PyRecfileObject* self, npy_intp col, char* ptr) {
    int status=1;
    fprintf(stderr,"implement bracketed arrays for postgres\n");
    status=0;
    return status;
}
static int
write_ascii_col(struct PyRecfileObject* self, npy_intp col, char* ptr) {
    int status=1;
    npy_intp nel=0, el=0, fsize=0;
    fsize = self->elsize[col];

    nel=self->nel[col];
    for (el=0; el<nel; el++) {
        status=write_ascii_col_element(self, col, ptr);
        if (status != 1) {
            break;
        }
        if (el < (nel-1)) {
            fprintf(self->fptr, "%s", self->delim);
        }
        ptr += fsize;
    }
    return status;
}

static int
write_ascii_row(struct PyRecfileObject* self, char* ptr) {
    int status=1;
    npy_intp col=0;

    for (col=0; col<self->ncols; col++) {
        if (self->bracket_arrays) {
            status=write_ascii_col_bracketed(self, col, ptr);
        } else {
            status=write_ascii_col(self, col, ptr);
        }
        if (status != 1) {
            break;
        }
        if (col < (self->ncols-1)) {
            fprintf(self->fptr, "%s", self->delim);
        }
        ptr += self->elsize[col]*self->nel[col];
    }
    return status;
}

static PyObject*
PyRecfileObject_write_ascii(struct PyRecfileObject* self, PyObject* args) 
{
    PyObject* array=NULL;
    char* ptr=NULL;
    npy_intp nrows=0, row=0;
    if (!PyArg_ParseTuple(args, (char*)"O", &array)) {
        return NULL;
    }

    if (!self->is_ascii) {
        PyErr_SetString(PyExc_IOError, "Don't use C write() method for binary");
        return NULL;
    }
    ptr = PyArray_DATA((PyArrayObject*) array);
    nrows = PyArray_SIZE((PyArrayObject*) array);

    for (row=0; row<nrows; row++) {
        if (!write_ascii_row(self, ptr)) {
            PyErr_Format(PyExc_IOError,"failed to write row %ld", row);
            return NULL;
        }
        fprintf(self->fptr, "\n");
        ptr += self->rowsize;
    }
    Py_RETURN_NONE;
}




static int
read_bytes(struct PyRecfileObject* self, npy_intp nbytes, char* buffer) 
{
    int status=1;
    if (1 != fread(buffer, nbytes, 1, self->fptr)) {
        PyErr_Format(PyExc_IOError, "failed to read col %ld bytes", 
                     nbytes);
        status=0;
    }
    return status;
}


static int
convert_value(struct PyRecfileObject* self, npy_intp col, const char *source, Py_ssize_t len, char *buffer)
{
   int status = 1;
   // Create string object for value to pass to user defined python function
   PyObject *sourceStr = PyString_FromStringAndSize(source, len);

   // Call user provided converter function
   PyObject *result = PyObject_CallFunction(self->converters[col], "S", sourceStr);
   if (result == 0) {
      PyObject *typeStr = PyObject_CallMethod(self->converters[col], "getTypeStr", NULL);
      PyErr_Format(PyExc_TypeError, "%s object is not callable", PyString_AsString(typeStr));
      Py_DECREF(typeStr);
      return 0;
   }
   Py_DECREF(sourceStr);

   // Output from converter function was converted to proper dtype.
   // Store value in array.
   PyArray_Descr *descr =  PyArray_DescrFromScalar(result);
   if (PyTypeNum_ISEXTENDED(descr->type_num)) {
      void *dataPtr = 0;
      PyArray_ScalarAsCtype(result, &dataPtr);
      if (dataPtr > 0) {
         memcpy(buffer, dataPtr, self->elsize[col]);
      }
   }
   else {
      PyArray_ScalarAsCtype(result, buffer);
   }
   Py_DECREF(result);

   return status;
}

static int
scan_number(struct PyRecfileObject* self, npy_intp col, char* buffer) {
    int status=1;
    int ret=0;
    char* fmt=NULL;

    fmt = PyArray_GETPTR1((PyArrayObject*)self->scan_formats_obj, col);

    ret = fscanf(self->fptr, fmt, buffer);
    if (ret != 1) {
        PyErr_Format(PyExc_IOError,
                "error reading number col %ld: fmt: '%s' ret: %d\n", 
                col, fmt, ret);
        status=0;
    }
    return status;
}

/*
 * Read an un-quoted variable length string.  We stop reading when we hit an
 * un-escaped delimiter or a newline.  The user should be careful when the
 * delimiter is a space; unescaped leading spaces will look like the delimiter
 * and must be escaped.
 *
 * Note because the row,column could be a var string and the user might not end
 */

static int
next_var_string_length(struct PyRecfileObject* self)
{
    int start = ftell(self->fptr);
    int len = 0;

    int status = 0;
    char cprev = 0;
    char c = fgetc(self->fptr);
    while ( (c != self->delim[0] || cprev == '\\') && c != '\n') {
        if (c == EOF) {
            PyErr_Format(PyExc_IOError, "Hit EOF looking for next variable length string\n");
            status=0;
            break;
        }

        len++;
        cprev = c;
        c = fgetc(self->fptr);
    }

    fseek(self->fptr, start, SEEK_SET);
    return len;
}

static int 
read_var_string(struct PyRecfileObject* self,
                   npy_intp nbytes, 
                   char* buffer)
{

    int status=1;
    char c=0, cprev=0;
    npy_intp istore=0;

    // read until we find an un-escaped delimiter, but only store
    // up to nbytes worth of data in buffer.

    c = fgetc(self->fptr);
    while ( (c != self->delim[0] || cprev == '\\') && c != '\n') {
        if (c == EOF) {
            PyErr_Format(PyExc_IOError, "Hit EOF extracting variable length string\n");
            status=0;
            break;
        }

        if (c == self->delim[0] && cprev == '\\') {
            // was escaped delim character
            buffer[istore-1] = c;
            // don't advance istore
        } else {
            if (istore < nbytes) {
                buffer[istore] = c;
                istore++;
            }
        }
        cprev=c;
        c = fgetc(self->fptr);
    }

    return status;
}


static int 
next_quoted_string_length(struct PyRecfileObject* self)
{
    int status = 1;
    npy_intp nread=0, istore=0;
    char c=0, cprev=0;
    int len=0;

    int start = ftell(self->fptr);

    // read until we find a non-whitespace character
    c = fgetc(self->fptr); nread++;
    while (c == ' ' || c == '\t') {
        if (c == EOF) {
            PyErr_Format(PyExc_IOError, "Hit EOF extracting quoted string\n");
            return 0;
        }
        c = fgetc(self->fptr); nread++;
    }

    if (c != self->quote_char[0]) {
        // quote char not found, treat as a variable length string
        // rewind before we started looking for quote
        fseeko(self->fptr, -nread, SEEK_CUR);
        //return read_var_string(self, nbytes, buffer);
        return next_var_string_length(self);
    }

    // read until we find the quote character, but only store
    // up to nbytes worth of data in buffer.  Allow escaped
    // quote characters using forward slash

    c = fgetc(self->fptr);
    while (c != self->quote_char[0] || cprev == '\\') {
        if (c == EOF) {
            PyErr_Format(PyExc_IOError, "Hit EOF extracting quoted string\n");
            return 0;
        }

        len++;
        cprev=c;
        c = fgetc(self->fptr);
    }

    fseek(self->fptr, start, SEEK_SET);
    return len;
}


/* 
 * The string may be quoted.  In this case we must treat the data as variable
 * length.  All characters are allowed within the string, even newlines.
 * Note the quote character can also appear if escaped, preceded by forward 
 * slash
 *
 * We first skip past any spaces or tabs
 *
 * If the first character we find is not the quote character, we proceed as if
 * this was a variable length string, in which newlines are not allowed.
 *
 * Only store as many bytes are are in the string (bytes parameter)
 *
 * if we do not encounter the quote character at the beginning, then we just
 * call the read_var_string code to look for the delimiter
 */

static int 
read_quoted_string(struct PyRecfileObject* self,
                   npy_intp nbytes, 
                   char* buffer)
{
    int status = 1;
    npy_intp nread=0, istore=0;
    char c=0, cprev=0;

    // read until we find a non-whitespace character
    c = fgetc(self->fptr); nread++;
    while (c == ' ' || c == '\t') {
        if (c == EOF) {
            PyErr_Format(PyExc_IOError, "Hit EOF extracting quoted string\n");
            status=0;
            goto _error_out_rdquote;
        }
        c = fgetc(self->fptr); nread++;
    }

    if (c != self->quote_char[0]) {
        // quote char not found, treat as a variable length string
        // rewind before we started looking for quote
        fseeko(self->fptr, -nread, SEEK_CUR);
        //return read_var_string(self, nbytes, buffer);

        if (self->has_var_strings) {
            // this will consume the delimiter or newline
            return read_var_string(self, nbytes, buffer);
        } else {
            // we must consume the delimiter or newline
            status = read_bytes(self, nbytes, buffer);
            fgetc(self->fptr);
            return status;
        }
    }

    // read until we find the quote character, but only store
    // up to nbytes worth of data in buffer.  Allow escaped
    // quote characters using forward slash

    c = fgetc(self->fptr);
    while (c != self->quote_char[0] || cprev == '\\') {
        if (c == EOF) {
            PyErr_Format(PyExc_IOError, "Hit EOF extracting quoted string\n");
            status=0;
            goto _error_out_rdquote;
        }
        
        if (c == self->quote_char[0] && cprev == '\\') {
            // was escaped quote character
            buffer[istore-1] = c;
            // don't advance istore
        } else {
            if (istore < nbytes) {
                buffer[istore] = c;
                istore++;
            }
        }
        cprev=c;
        c = fgetc(self->fptr);
    }

    // done reading the string, now make sure we consume the delimiter
    c = fgetc(self->fptr);
    while (c != self->delim[0] && c != '\n') {
        if (c == EOF) {
            PyErr_Format(PyExc_IOError, "Hit EOF finding delim/newline\n");
            status=0;
            goto _error_out_rdquote;
        }
        c = fgetc(self->fptr);
    }

_error_out_rdquote:

    return status;
}


// consume the data and the delimiter (or newline)
// note number columns automatically consume the delimiter
static int
read_ascii_col_element(struct PyRecfileObject* self,
                         npy_intp col,
                         char* buffer) 
{
    int status=1;
    npy_intp typenum=0;
    PyObject *converter=0;

    typenum=self->typenums[col];
    converter=self->converters[col];

    if (converter != Py_None) {
        if (self->has_quoted_strings) {
            int len = next_quoted_string_length(self);
            char *temp = calloc(len, 1);
            status = read_quoted_string(self, len, temp);
            status = convert_value(self, col, temp, len, buffer);
            free(temp);
        }
        // Since we're using a user defined function to convert
        // string to dtype instead of fscanf, treat string as
        // variable length if converting to non string dtype.
        else if (self->has_var_strings || NPY_STRING != typenum) {
            int len = next_var_string_length(self);
            char *temp = calloc(len, 1);
            status = read_var_string(self, len, temp);
            status = convert_value(self, col, temp, len, buffer);
            free(temp);
        }
        else {
            int len = self->elsize[col];
            char *temp = calloc(len, 1);
            status = read_bytes(self, len, temp);
            status = convert_value(self, col, temp, len, buffer);
            free(temp);

            // consume delimiter/newline
            char c = fgetc(self->fptr);
            if (c != self->delim[0] && c != '\n') {
                fprintf(stderr,"col %ld tried to read delim '%s' or newline, but got '%c'\n", 
                col, self->delim,c);
            }
        }
    }
    else if (NPY_STRING == typenum) {
        if (self->has_quoted_strings) {
            status = read_quoted_string(self, self->elsize[col], buffer);
        } else if (self->has_var_strings) {
            status = read_var_string(self, self->elsize[col], buffer);
        } else {
            char c;
            // read as fixed width bytes
            status = read_bytes(self, self->elsize[col], buffer);
            // we must consume the delimiter/newline
            c = fgetc(self->fptr);
            if (c != self->delim[0] && c != '\n') {
                fprintf(stderr,"col %ld tried to read delim '%s' or newline, but got '%c'\n", 
                col, self->delim,c);
            }
        }
    } else {
        status = scan_number(self, col, buffer);

        // our scan formats will consume the delimiter when it is not a space,
        // but if it is a space we have to consume it manually
        if (self->delim_is_space) {
            fgetc(self->fptr);
        }
    }
    return status;
}

/*
 * skip=0 means don't increment the buffer
 */
static int
read_ascii_col(
        struct PyRecfileObject* self, npy_intp col, char* buffer, int skip) 
{
    int status=1;
    npy_intp nel=0, el=0, fsize=0;

    fsize = self->elsize[col];

    nel=self->nel[col];

    for (el=0; el<nel; el++) {
        status=read_ascii_col_element(self, col, buffer);
        if (status != 1) {
            break;
        }

        if (!skip) {
            // if skipping, just re-use buffer
            buffer += fsize;
        }
    }
    return status;
}


static int
is_comment_or_blank_line(struct PyRecfileObject *self)
{
    int status = 0;
    char c = 0;

    int start = ftell(self->fptr);

    c = fgetc(self->fptr);
    while (c == ' ' || c == '\t') {
        c= fgetc(self->fptr);
    }

    if (c == '\n' || c == EOF)
    {
        status = 1;
    }
    else if (self->comment_char != NULL)
    {
        if (c == self->comment_char[0])
        {
            status = 1;
        }
    }
    
    fseek(self->fptr, start, SEEK_SET);

    return status;
}


static int
read_ascii_row(struct PyRecfileObject* self, char* ptr, int skip) {
    int status=1;
    npy_intp col=0;

    while (is_comment_or_blank_line(self) || feof(self->fptr))
    {
        // skip to next line
        char c = fgetc(self->fptr);
        while (c != '\n' && c != EOF) {
            c = fgetc(self->fptr);
        }

        if (c == EOF)
        {
            return 0;
        }
    }

    for (col=0; col<self->ncols; col++) {
        status=read_ascii_col(self, col, ptr, skip);
        if (status != 1) {
            break;
        }
        if (!skip) {
            // we were just reading into a buffer
            ptr += self->elsize[col]*self->nel[col];
        }
    }

    return status;
}

static int
skip_ascii_rows(struct PyRecfileObject* self, npy_intp nrows) {
    int status=1;
    npy_intp i=0;
    int skip=1;
    for (i=0; i<nrows; i++) {
        status=read_ascii_row(self, self->buffer, skip);
        if (status != 1) {
            break;
        }
    }
    return status;
}

static PyObject* 
read_ascii_slice(struct PyRecfileObject* self,
                 void* ptr, npy_intp start, npy_intp stop, npy_intp step) {

    int skip=0;
    npy_intp current_row=0, row=0;

    current_row=0;
    row=start;
    while (row < stop) {
        if (row > current_row) {
            if (!skip_ascii_rows(self, row-current_row)) {
                return NULL;
            }
            current_row=row;
        }

        if (!read_ascii_row(self, ptr, skip)) {
            return NULL;
        }

        ptr += self->rowsize;
        current_row += 1;
        row += step;
    }
    Py_RETURN_NONE;

}

static PyObject* 
read_binary_slice(struct PyRecfileObject* self,
                  void* ptr, npy_intp start, npy_intp stop, npy_intp step) {

    npy_intp current_row=0, row=0;
    off_t bytes=0;

    current_row=0;
    row=start;
    while (row < stop) {
        if (row > current_row) {
            bytes = (row-current_row)*self->rowsize;
            if (0 != fseeko(self->fptr, bytes , SEEK_CUR)) {
                PyErr_SetString(PyExc_IOError, "failed to seek");
                return NULL;
            }
            current_row=row;
        }

        if (1 != fread(ptr, (size_t)self->rowsize, 1, self->fptr)) {
            PyErr_Format(PyExc_IOError, "failed to read row %ld", row);
            return NULL;
        }

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
        return read_ascii_slice(self, ptr, start, stop, step);
    } else {
        return read_binary_slice(self, ptr, start, stop, step);
    }
}

/*
 * Count ascii rows using full reading code.  This works even when there are
 * newlines in the strings, but is much slower than just counting newlines.
 */

static PyObject* 
PyRecfileObject_count_ascii_rows(struct PyRecfileObject* self, PyObject* args)
{
    PY_LONG_LONG nrows=0;
    while (1 == read_ascii_row(self, self->buffer, 1)) {
        nrows++;
    }
    PyErr_Clear();
    return PyLong_FromLongLong(nrows);
}



static int
read_ascii_cols(struct PyRecfileObject* self, 
                char* ptr, 
                npy_intp* cols, 
                npy_intp ncols)
{
    int status=1;
    npy_intp icol=0, current_col=0, col=0;
    npy_intp tcol=0;

    for (icol=0; icol<ncols; icol++) {

        if (ncols < self->ncols) {
            col=cols[icol];
        } else {
            col=icol;
        }

        if (col > current_col) {
            // skip some columns
            for (tcol=current_col; tcol<col; tcol++) {
                status=read_ascii_col(self, tcol, self->buffer, 1);
                if (status != 1) {
                    return status;
                }
            }
            current_col=col;
        }

        status=read_ascii_col(self, col, ptr, 0);
        if (status != 1) { 
            return status;
        }

        ptr += self->elsize[col]*self->nel[col];
        current_col++;
    }

    // skip remaining columns if necessary
    if (col < (self->ncols-1)) {
        for (tcol=col+1; tcol<self->ncols; tcol++) {
            status=read_ascii_col(self, tcol, self->buffer, 1);
            if (status != 1) {
                return status;
            }
        }
    }

    return status;

}

static PyObject* 
read_ascii_subset(struct PyRecfileObject* self,
                  void* ptr, 
                  npy_intp itemsize,
                  npy_intp* cols, npy_intp ncols,
                  npy_intp* rows, npy_intp nrows) 
{
    int status=1;
    npy_intp current_row=0, row=0, irow=0;
    int skip=0;

    for (irow=0; irow<nrows; irow++) {

        if (nrows < self->nrows) {
            row=rows[irow];
        } else {
            row=irow;
        }

        if (row > current_row) {
            if (!skip_ascii_rows(self, row-current_row)) {
                return NULL;
            }
            current_row=row;
        }

        if (ncols < self->ncols) {
            status=read_ascii_cols(self, ptr, cols, ncols);
        } else {
            status=read_ascii_row(self, ptr, skip);
        }
        if (status!=1) {
            return NULL;
        }

        ptr += itemsize;
        current_row++;
    }


    Py_RETURN_NONE;
}

static int
read_binary_cols(struct PyRecfileObject* self, 
                char* ptr, 
                npy_intp* cols, 
                npy_intp ncols)
{
    npy_intp icol=0, current_col=0, col=0;
    off_t bytes=0;

    for (icol=0; icol<ncols; icol++) {

        if (ncols < self->ncols) {
            col=cols[icol];
        } else {
            col=icol;
        }

        if (col > current_col) {
            bytes = self->offset[col]-self->offset[current_col];
            if (0 != fseeko(self->fptr, bytes , SEEK_CUR)) {
                PyErr_SetString(PyExc_IOError, "failed to seek");
                return 0;
            }
            current_col=col;
        }

        bytes = self->elsize[col]*self->nel[col];
        if (1 != fread(ptr, bytes, 1, self->fptr)) {
            PyErr_Format(PyExc_IOError, "failed to read col %ld", col);
            return 0;
        }

        ptr += self->elsize[col]*self->nel[col];
        current_col++;
    }

    // skip remaining columns if necessary
    if (col < (self->ncols-1)) {
        bytes = self->rowsize-self->offset[col+1];
        if (0 != fseeko(self->fptr, bytes , SEEK_CUR)) {
            PyErr_SetString(PyExc_IOError, "failed to seek");
            return 0;
        }
    }

    return 1;

}


static PyObject* 
read_binary_subset(struct PyRecfileObject* self,
                   void* ptr, 
                   npy_intp itemsize,
                   npy_intp* cols, npy_intp ncols,
                   npy_intp* rows, npy_intp nrows) 
{
    npy_intp current_row=0, row=0, irow=0;
    off_t bytes=0;

    for (irow=0; irow<nrows; irow++) {

        if (nrows < self->nrows) {
            row=rows[irow];
        } else {
            row=irow;
        }

        if (row > current_row) {
            bytes = (row-current_row)*self->rowsize;
            if (0 != fseeko(self->fptr, bytes , SEEK_CUR)) {
                PyErr_SetString(PyExc_IOError, "failed to seek");
                return NULL;
            }
            current_row=row;
        }

        if (ncols < self->ncols) {
            if (1 != read_binary_cols(self, ptr, cols, ncols)) {
                return NULL;
            }
        } else {
            if (1 != fread(ptr, (size_t)self->rowsize, 1, self->fptr)) {
                PyErr_Format(PyExc_IOError, "failed to read row %ld", row);
                return NULL;
            }
        }

        ptr += itemsize;
        current_row++;
    }

    Py_RETURN_NONE;
}




/* 
 * read specific rows and columns into input array.  
 *
 * cols and rows should be sorted, unique npy_intp arrays
 * or None
 *
 * If cols are None, all cols are read.
 * If rows are None, all rows are read.
 * If both are None, you should really use read_slice!
 *
 * 
 * As usual, most error checking must be done in the python wrapper!  The
 * exception is matching the itemsize of the input array to the rowsize in
 * order to avoid out of bounds errors.
 */

static PyObject* 
PyRecfileObject_read_subset(struct PyRecfileObject* self, PyObject* args)
{
    PyObject* array=NULL;
    PyObject* rows_obj=NULL;
    PyObject* cols_obj=NULL;
    npy_intp nrows=0;
    npy_intp* rows=NULL;
    npy_intp ncols=0;
    npy_intp* cols=NULL;
    npy_intp itemsize=0;
    void* ptr=NULL;
    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &array,
                          &cols_obj,
                          &rows_obj)) {
        return NULL;
    }

    if (cols_obj != Py_None) {
        cols = (npy_intp*) PyArray_DATA((PyArrayObject*)cols_obj);
        ncols=PyArray_SIZE((PyArrayObject*)cols_obj);
    } else {
        ncols=self->ncols;
    }
    if (rows_obj != Py_None) {
        rows = (npy_intp*) PyArray_DATA((PyArrayObject*)rows_obj);
        nrows=PyArray_SIZE((PyArrayObject*)rows_obj);
    } else {
        nrows=self->nrows;
    }


    ptr = PyArray_DATA((PyArrayObject*) array);
    itemsize = PyArray_ITEMSIZE((PyArrayObject*) array);
    if (self->is_ascii) {
        return read_ascii_subset(self, ptr, itemsize, cols, ncols, rows, nrows);
    } else {
        return read_binary_subset(self, ptr, itemsize, cols, ncols, rows, nrows);
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

    free(self->buffer);

    self->fptr=NULL;
    self->file_obj=NULL;
    Py_XDECREF(self->elsize_obj);
    Py_XDECREF(self->nel_obj);
    Py_XDECREF(self->offset_obj);
    Py_XDECREF(self->scan_formats_obj);
    Py_XDECREF(self->print_formats_obj);

    self->elsize_obj=NULL;
    self->elsize=NULL;
    self->nel_obj=NULL;
    self->nel=NULL;
    self->offset_obj=NULL;
    self->offset=NULL;
    self->scan_formats_obj=NULL;
    self->print_formats_obj=NULL;
    self->buffer=NULL;
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
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_INT8), PyString_FromString(NPY_INT8_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_UINT8), PyString_FromString(NPY_UINT8_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_INT16), PyString_FromString(NPY_INT16_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_UINT16), PyString_FromString(NPY_UINT16_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_INT32), PyString_FromString(NPY_INT32_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_UINT32), PyString_FromString(NPY_UINT32_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_INT64), PyString_FromString(NPY_INT64_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_UINT64), PyString_FromString(NPY_UINT64_FMT));

#ifdef NPY_INT128
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_INT128), PyString_FromString(NPY_INT128_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_UINT128), PyString_FromString(NPY_UINT128_FMT));
#endif
#ifdef NPY_INT256
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_INT256), PyString_FromString(NPY_INT256_FMT));
    PyDict_SetItem(dict, PyLong_FromLong((long)NPY_UINT256), PyString_FromString(NPY_UINT256_FMT));
#endif

    if (type == 1) {
        // scan formats
        PyDict_SetItem(dict, PyLong_FromLong((long)NPY_FLOAT32), PyString_FromString("f"));
        PyDict_SetItem(dict, PyLong_FromLong((long)NPY_FLOAT64), PyString_FromString("lf"));

#ifdef NPY_FLOAT128
        PyDict_SetItem(dict, PyLong_FromLong((long)NPY_FLOAT128), PyString_FromString("Lf"));
#endif
    } else {
        // print formats
        PyDict_SetItem(dict, PyLong_FromLong((long)NPY_FLOAT32), PyString_FromString(".7g"));
        PyDict_SetItem(dict, PyLong_FromLong((long)NPY_FLOAT64), PyString_FromString(".16g"));

#ifdef NPY_FLOAT128
        // what should this be?
        PyDict_SetItem(dict, PyLong_FromLong((long)NPY_FLOAT128), PyString_FromString(".16g"));
#endif

        PyDict_SetItem(dict, PyLong_FromLong((long)NPY_STRING), PyString_FromString("s"));
    }
    return dict;
}


static PyMethodDef PyRecfileObject_methods[] = {
    {"set_nrows", (PyCFunction)PyRecfileObject_set_nrows,  METH_VARARGS,  "Set the nrows field\n"},
    {"write_ascii", (PyCFunction)PyRecfileObject_write_ascii,  METH_VARARGS,  "Write the input array to ascii\n"},
    {"read_slice",  (PyCFunction)PyRecfileObject_read_slice,   METH_VARARGS,  "read a slice into input array. Use for step != 1, otherwise can use numpy.fromfile\n"},
    {"read_subset", (PyCFunction)PyRecfileObject_read_subset,  METH_VARARGS,  "read arbitrary subsets\n"},
    {"count_ascii_rows", (PyCFunction)PyRecfileObject_count_ascii_rows,  METH_VARARGS,  "count rows in ascii file\n"},
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
