"""
numpy.recfile

This module defines the Recfile class for reading and writing structured arrays
to and from files.  Files can be either binary or text files.

See docs for numpy.recfile.Recfile for more details.

    examples
    ---------
        # read from binary file
        dtype=[('id','i4'),('x','f8'),('y','f8'),('arr','f4',(2,2))]
        rec=numpy.recfile.Recfile(fname,dtype=dtype)


        # read all data using either slice or method notation
        data=rec[:]
        data=rec.read()

        # read row slices
        data=rec[8:55:3]

        # read subset of columns and possibly rows
        # can use either slice or method notation
        data=rec['x'][:]
        data=rec['x','y'][:]
        data=rec[col_list][row_list]
        data=rec.read(columns=col_list, rows=row_list)

        # for text files, just send the delimiter string
        # all the above calls will also work
        rec=numpy.recfile.Recfile(fname,dtype=dtype,delim=',')

        # save time for text files by sending row count
        rec=numpy.recfile.Recfile(fname,dtype=dtype,delim=',',nrows=10000)

        # write some data
        rec=numpy.recfile.Recfile(fname,mode='w',dtype=dtype,delim=',')
        rec.write(data)

        # append some data
        rec.write(more_data)

        # print metadata about the file
        print rec
        Recfile  nrows: 345472 ncols: 6 mode: 'w'

          id                 <i4
          x                  <f8
          y                  <f8
          arr                <f4  array[2,2]
"""
import os
import sys
from sys import stderr
import numpy
import unittest
import tempfile
import _recfile

class Recfile(_recfile.Recfile):
    """
    A class for reading and writing structured arrays to and from files.

    Both binary and text files are supported.  Any subset of the data can be
    read without loading the whole file.  See the limitations section below for
    caveats.

    parameters
    ----------
    fobj: file or string
        A string or file object.
    mode: string
        Mode for opening when fobj is a string 
    dtype:
        A numpy dtype or descriptor describing each line of the file.  The
        dtype must contain fields. This is a required parameter; it is a
        keyword only for clarity. 

        Note for text files the dtype will be converted to native byte
        ordering.  Any data written to the file must also be in the native byte
        ordering.
    nrows: int, optional
        Number of rows in the file.  If not entered, the rows will be counted
        from the file itself. This is a simple calculation for binary files,
        but can be slow for text files.
    delim: string, optional
        The delimiter for text files.  If None or "" the file is
        assumed to be binary.  Should be a single character.
    skipheader: int, optional
        Skip this many lines in the header.
    offset: int, optional
        Move to this offset in the file.  Reads will all be relative to this
        location. If not sent, it is taken from the current position in the
        input file object or 0 if a filename was entered.

    string_newlines: bool, optional
        If true, strings in text files may contain newlines.  This is only
        relevant for text files when the nrows= keyword is not sent, because
        the number of lines must be counted.  
        
        In this case the full text reading code is used to count rows instead
        of a simple newline count.  Because the text is fully processed twice,
        this can double the time to read files.

        if quote_char is sent, the same slower line count mechanism is also
        used.

    quote_char: string, optional
        The text file to be read may have quoted strings.  This character is
        used in the file to quote strings, e.g.  "my string".  The string *can*
        contain the quote character itself if escaped, e.g.

            "my string has a \\" in it"

        The quote_char must be a length 1 string or ""

    var_strings: bool, optional

        The text file to be read may have unquoted variable length strings.
        Note if the string in the file is larger than allowed by the input
        dtype describing this column, then the result will be truncated.

        The delimiter may appear in the string as long as it is escaped, 
        e.g. for comma delimited \\,

        Note variable length strings cannot contain the newline character, as
        it is effectively the delimiter for the last column in a row.  If you
        need to have newlines in the strings, use a quoted string or a fixed
        width string

    padnull: bool
        If True, nulls in strings are replaced with spaces when writing text
    ignorenull: bool
        If True, nulls in strings are not written when writing text.  This
        results in string fields that are not fixed width, so cannot be
        read back in using recfile

    limitations
    -----------
        Currently, only fixed width string fields are supported.  String fields
        can contain any characters, including newlines, but for text files
        quoted strings are not currently supported: the quotes will be part of
        the result.  For binary files, structured sub-arrays and complex can be
        written and read, but this is not supported yet for text files. 

    examples
    ---------
        # read from binary file
        dtype=[('id','i4'),('x','f8'),('y','f8'),('arr','f4',(2,2))]
        rec=numpy.recfile.Recfile(fname,dtype=dtype)


        # read all data using either slice or method notation
        data=rec[:]
        data=rec.read()

        # read row slices
        data=rec[8:55:3]

        # read subset of columns and possibly rows
        # can use either slice or method notation
        data=rec['x'][:]
        data=rec['x','y'][:]
        data=rec[col_list][row_list]
        data=rec.read(columns=col_list, rows=row_list)

        # for text files, just send the delimiter string
        # all the above calls will also work
        rec=numpy.recfile.Recfile(fname,dtype=dtype,delim=',')

        # save time for text files by sending row count
        rec=numpy.recfile.Recfile(fname,dtype=dtype,delim=',',nrows=10000)

        # write some data
        rec=numpy.recfile.Recfile(fname,mode='w',dtype=dtype,delim=',')
        rec.write(data)

        # append some data
        rec.write(more_data)

        # print metadata about the file
        print rec
        Recfile  nrows: 345472 ncols: 6 mode: 'w'

          id                 <i4
          x                  <f8
          y                  <f8
          arr                <f4  array[2,2]
    """
    def __init__(self, fobj, mode='r', dtype=None, **keys):

        if dtype is None:
            raise ValueError("You must enter a dtype for each row")

        if isinstance(fobj, file):
            self.fobj_was_input = True
            self.fobj = fobj
        elif isinstance(fobj, basestring):
            self.fobj_was_input = False
            fpath = os.path.expanduser(fobj)
            fpath = os.path.expandvars(fpath)
            if mode == 'r+' and not os.path.exists(fpath):
                # path doesn't exist but we want to append.  Change the
                # mode to w+
                mode = 'w+'
            self.fobj = open(fpath, mode)
        else:
            raise ValueError("Input file should be string or file object")

        self.dtype=numpy.dtype(dtype)
        if self.dtype.names is None:
            raise ValueError("dtype must have fields")

        self.delim = keys.get('delim',None)
        self.padnull=keys.get('padnull',False)
        self.ignorenull=keys.get('ignorenull',False)
        self.skipheader = keys.get('skipheader',None)
        # this can be input, in which case we will
        # seek to this offset for reads
        self.file_offset=keys.get('offset',None)
        self.string_newlines=keys.get('string_newlines',False)


        self.quote_char = keys.get('quote_char','')
        if self.quote_char is None:
            self.quote_char = ""
        if len(self.quote_char) > 1:
            raise ValueError("quote char must have len <= 1")
        self.var_strings = keys.get('var_strings',False)

        self.padnull = 1 if self.padnull else 0
        self.ignorenull = 1 if self.ignorenull else 0
        self.var_strings = 1 if self.var_strings else 0

        if self.file_offset is None:
            self.file_offset = self.fobj.tell()
        else:
            if self.fobj.tell() != self.file_offset:
                self.fobj.seek(self.file_offset)

        if self.delim is None:
            self.delim=""
        if self.delim != "":
            self.is_ascii = 1
            self._nativize_dtype()
        else:
            self.is_ascii = 0

        self._set_possible_formats()
        self._set_col_info()

        if self.skipheader is not None:
            self._skipheader_lines(self.skipheader)


        super(Recfile, self).__init__(self.fobj,
                                      self.delim,
                                      self.is_ascii,
                                      self.typenums,
                                      self.elsize,
                                      self.nel,
                                      self.offset,
                                      self.scan_formats,
                                      self.print_formats,
                                      self.padnull,
                                      self.ignorenull,
                                      self.quote_char,
                                      self.var_strings)
        self._set_beginning_nrows(**keys)

    def write(self, data, **keys):
        """
        Write the input structured array to the file

        parameters
        ----------
        data: structured array
            A numpy array with fields defined.  The dtype must match that of
            the dtype parameter passed when constructing the Recfile object.

            Note that for text files, the dtype passed on construction was
            converted to native byte ordering, and this array must also have
            native byte ordering.
        """
        if self.fobj.closed:
            raise ValueError("file is not open")
        if self.fobj.mode[0] != 'w' and '+' not in self.fobj.mode:
            raise ValueError("You must open with 'w*' or 'r+' to write")
        if data.dtype != self.dtype:
            raise ValueError("Input dtype:\n\t%s\n"
                             "does not match file:\n\t%s" % (data.dtype.descr,self.dtype.descr))

        dataview = data.view(numpy.ndarray) 
        if self.delim == "":
            dataview.tofile(self.fobj)
        else:
            super(Recfile,self).write_ascii(dataview)


        self.fobj.flush()
        self.nrows += dataview.size

        nrows_send = numpy.array([self.nrows],dtype='intp')
        super(Recfile,self).set_nrows(nrows_send)

    def read(self, rows=None, columns=None, **keys):
        """
        Read data from the file.

        A subset of the data can be read by sending rows= or columns=

        parameters
        ----------
        rows: optional
            A number or sequence/array of numbers representing a subset of the
            rows to read.
        columns: optional
            A string or sequence/array of strings representing a subset of the
            columns to read.
        """
        if self.fobj.closed:
            raise ValueError("file is not open")
        if self.fobj.mode[0] != 'r' and '+' not in self.fobj.mode:
            raise ValueError("You must open with 'r*' or 'w+' to read")

        if self.fobj.tell() != self.file_offset:
            self.fobj.seek(self.file_offset)

        if rows is None and columns is None:
            return self._read_all()
        else:
            return self._read_subset(columns=columns, rows=rows)


    def close(self):
        """
        Close the file.  If a file object was sent on construction, it is not
        closed.
        """
        if not self.fobj_was_input:
            self.fobj.close()

    def _read_all(self):
        """
        Read all rows and fields
        """

        if self.delim == "":
            data = numpy.fromfile(self.fobj,dtype=self.dtype,count=self.nrows)
        else:
            return self[:]
        return data

    def _read_subset(self, columns=None, rows=None):
        cols2read = self._get_cols2read(columns)
        rows2read = self._get_rows2read(rows)
        if cols2read is not None:
            dtype=self._get_dtype_subset(cols2read)
        else:
            dtype=self.dtype

        if rows2read is not None:
            nrows=rows2read.size
        else:
            nrows=self.nrows

        result = numpy.zeros(nrows, dtype=dtype)
        super(Recfile,self).read_subset(result, cols2read, rows2read)

        if len(dtype.names) == 1:
            result=result[dtype.names[0]]
        return result

    def _read_slice(self, slc):
        """
        error checking should happen elsewhere
        """
        if self.fobj.closed:
            raise ValueError("file is not open")
        if self.fobj.mode[0] != 'r' and '+' not in self.fobj.mode:
            raise ValueError("You must open with 'r*' or 'w+' to read")

        if self.fobj.tell() != self.file_offset:
            self.fobj.seek(self.file_offset)

        if self.delim == "" and slc.step==1:
            result = self._read_binary_range(slc.start, slc.stop)
        else:
            result = self._read_slice_any(slc)
        return result

    def _read_slice_any(self, slc):
        """
        Read row slice, stop inclusive

        This is split off because of special case of contiguous
        binary ranges.

        slice error checking should happen before here; also resetting
        to file offset start beforehand.
        """

        nrows = self._slice_nrows(slc)
        rowsize=self.dtype.itemsize
        result = numpy.zeros(nrows, dtype=self.dtype)
        try:
            self.read_slice(result, 
                            numpy.array([slc.start],dtype='intp'),
                            numpy.array([slc.stop], dtype='intp'),
                            numpy.array([slc.step], dtype='intp'))
        except:
            raise
        return result

    def _read_binary_range(self, start, stop):
        """
        Read contiguous range of rows, stop inclusive

        error checking should happen before here; also resetting
        to file offset start beforehand.
        """
        nrows=stop-start
        rowsize=self.dtype.itemsize
        if start > 0:
            self.fobj.seek(self.file_offset + rowsize*start)
        result = numpy.fromfile(self.fobj,dtype=self.dtype,count=nrows)
        return result

    def _slice_nrows(self, slc):
        rdiff = slc.stop-slc.start
        extra=0
        if (rdiff % slc.step) != 0:
            extra = 1

        nrows = (rdiff//slc.step) + extra
        return nrows

    def __getitem__(self, arg):
        """
        sf = Recfile(....)

        # read subsets of columns and/or rows from the file.  Rows and
        # columns can be lists/tuples/arrays

        # read subsets of rows
        data = sf[:]
        data = sf[ 35 ]
        data = sf[ 35:88 ]
        data = sf[ [3,234,5551,.. ] ]

        # read subsets of columns
        data = sf['fieldname'][:]
        data = sf[ ['field1','field2',...] ][row_list]


        # read subset of rows *and* columns.
        data = sf['fieldname'][3:58]
        data = sf[fieldlist][rowlist]

        # Note, if you send just columns, a RecfileColumnSubset object is
        # returned
        sub = sf['fieldname']
        data = sub.read(rows=)
        """

        if self.fobj.closed:
            raise ValueError("file is not open")

        res, isrows, isslice = self._process_args_as_rows_or_columns(arg)
        if isrows:
            # rows were entered: read all columns
            if isslice:
                return self._read_slice(res)
            else:
                return self.read(rows=res)

        # columns was entered.  Return a subset objects
        return RecfileColumnSubset(self, columns=res)

    def _process_args_as_rows_or_columns(self, arg, unpack=False):
        """

        args must be a tuple.  Only the first one or two args are used.

        We must be able to interpret the args as as either a column name or
        row number, or sequences thereof.  Numpy arrays and slices are also
        fine.

        Examples:
            'field'
            35
            [35,55,86]
            ['f1',f2',...]
        Can also be tuples or arrays.

        """

        isslice = False
        isrows = False
        result=arg
        if isinstance(arg, (tuple,list,numpy.ndarray)):
            # a sequence was entered
            if isinstance(arg[0],basestring):
                pass
            else:
                isrows=True
                result = arg
        elif isinstance(arg,basestring):
            # a single string was entered
            pass
        elif isinstance(arg, slice):
            isrows=True
            isslice=True
            if unpack:
                result = self._slice2rows(arg.start, arg.stop, arg.step)
            else:
                result = self._process_slice(arg)
        else:
            # a single object was entered.  Probably should apply some more 
            # checking on this
            isrows=True

        return result, isrows, isslice

    def _process_slice(self, arg):
        start = arg.start
        stop = arg.stop
        step = arg.step

        if step is None:
            step=1
        if start is None:
            start = 0
        if stop is None:
            stop = self.nrows

        if start < 0:
            start = self.nrows + start
            if start < 0:
                raise IndexError("Index out of bounds")

        # remember, stop is inclusive
        if stop < 0:
            stop = self.nrows + stop

        if stop < start:
            # will return an empty struct
            stop = start
        if stop > self.nrows:
            stop=self.nrows

        return slice(start, stop, step)

    def _slice2rows(self, start, stop, step=None):
        if start is None:
            start=0
        if stop is None:
            stop=self.nrows
        if step is None:
            step=1

        tstart = self._fix_range(start)
        tstop  = self._fix_range(stop)
        if tstart == 0 and tstop == self.nrows:
            # this is faster: if all fields are also requested, then a 
            # single fread will be done
            return None
        if stop < start:
            raise ValueError("start is greater than stop in slice")
        return numpy.arange(tstart, tstop, step, dtype='intp')

    def _fix_range(self, num, isslice=True):
        """
        If el=True, then don't treat as a slice element
        """

        if isslice:
            # include the end
            if num < 0:
                num=self.nrows + (1+num)
            elif num > self.nrows:
                num=self.nrows
        else:
            # single element
            if num < 0:
                num=self.nrows + num
            elif num > (self.nrows-1):
                num=self.nrows-1

        return num

    def _get_rows2read(self, rows):
        if rows is None:
            return None
        try:
            # a sequence entered
            rowlen=len(rows)

            # no copy is made if it is an intp numpy array
            rows2read = numpy.array(rows,ndmin=1,copy=False, dtype='intp')
            if rows2read.size == 1:
                rows2read[0] = self._fix_range(rows2read[0], isslice=False)
        except:
            # single object entered
            rows2read = self._fix_range(rows, isslice=False)
            rows2read = numpy.array([rows2read], dtype='intp')

        # gets unique and sorted
        rows2read = numpy.unique(rows2read)
        rmin = rows2read[0]
        rmax = rows2read[-1]
        if rmin < 0 or rmax >= self.nrows:
            raise ValueError("Requested rows range from %s->%s: out of "
                             "range %s->%s" % (rmin,rmax,0,self.nrows-1))
        return rows2read

    def _get_cols2read(self, columns):

        if columns is None:
            return None

        cols = numpy.array(columns,ndmin=1,copy=False)
        if not isinstance(cols[0], basestring):
            raise ValueError("columns must be specified by name")

        cols2read = numpy.zeros(cols.size,dtype='intp')
        for i,col in enumerate(cols):
            w,=numpy.where(self.colnames == col)
            if w.size == 0:
                raise ValueError("colname not found: %s" % col)
            cols2read[i] = w[0]

        cols2read = numpy.unique(cols2read)
        cmin=cols2read[0]
        cmax=cols2read[-1]
        if cmin < 0 or cmax > (self.ncols-1):
            raise ValueError("Requested cols range from %s->%s: out of "
                             "range %s->%s" % (cmin,cmax,0,self.ncols-1))
        return cols2read

    def _get_dtype_subset(self, colnums):
        d=self.dtype.descr
        descr = [d[c] for c in colnums]
        return numpy.dtype(descr)

    def _set_col_info(self):
        """
        Get column info from the dtype object
        """
        nf=len(self.dtype.names)
        self.ncols = nf
        self.colnames = numpy.array(self.dtype.names)
        self.scan_formats=numpy.zeros(nf, dtype='S7')
        self.print_formats=numpy.zeros(nf, dtype='S7')
        self.typenums=numpy.zeros(nf, dtype='intp')
        self.elsize=numpy.zeros(nf, dtype='intp')
        self.nel=numpy.zeros(nf, dtype='intp')
        self.offset=numpy.zeros(nf, dtype='intp')

        for i,name in enumerate(self.dtype.names):
            self.offset[i] = self.dtype.fields[name][1]

            dt = self.dtype.fields[name][0]
            typenum = dt.base.num
            self.typenums[i] = typenum
            self.elsize[i] = dt.base.itemsize
            self.nel[i] = dt.itemsize/dt.base.itemsize


            if self.is_ascii:
                # we don't use scanf on strings
                if dt.base.char != 'S':
                    if typenum not in self.allscanf:
                        raise ValueError("scan type not yet "
                                         "supported: %s" % dt.base.str)
                    self.scan_formats[i] = self.allscanf[typenum]
                if typenum not in self.allprintf:
                    raise ValueError("print type not yet "
                                     "supported: %s" % dt.base.str)
                self.print_formats[i] = self.allprintf[typenum]

    def _set_possible_formats(self):
        # first set up all possible scan formats
        allscanf = _recfile.get_formats(1)
        allprintf = _recfile.get_formats(2)
        for d in allscanf:
            allscanf[d] = '%' + allscanf[d]
        for d in allprintf:
            allprintf[d] = '%' + allprintf[d]

        # if delim and not whitespace, we need to add
        # the delimiter to the scan format
        if len(self.delim) > 0:
            if self.delim[0] != " ":
                for d in allscanf:
                    allscanf[d] += " "+self.delim

        self.allscanf=allscanf
        self.allprintf=allprintf


    def _skipheader_lines(self, nlines):
        for i in xrange(nlines):
            tmp = self.fobj.readline()

        # now, we override any existing offset to our
        # position after skipping lines
        self.file_offset = self.fobj.tell()

    def _set_beginning_nrows(self, **keys):
        if self.fobj.mode[0] == 'w':
            self.nrows=0
        else:
            self.nrows = keys.get('nrows',-1)
            if self.nrows is None:
                self.nrows = -1

            if self.nrows < 0 and self.fobj.mode[0] == 'r':
                self.nrows = self._get_nrows()
        self._set_nrows(self.nrows)
            
    def _set_nrows(self, nrows):
        nrows_send = numpy.array([nrows],dtype='intp')
        super(Recfile,self).set_nrows(nrows_send)

    def _get_nrows(self):
        """
        Count rows from beginning current offset
        """
        import time
        if self.delim != "":
            if self.fobj.tell() != self.file_offset:
                self.fobj.seek(self.file_offset)
            if self.string_newlines:
                nrows=super(Recfile, self).count_ascii_rows()
            else:
                nrows = 0
                for line in self.fobj:
                    nrows += 1

            self.fobj.seek(self.file_offset)
        else:
            # For binary, try to figure out the number of rows based on
            # the number of bytes past the offset

            rowsize=self.dtype.itemsize
            # go to end
            self.fobj.seek(0,2)
            datasize = self.fobj.tell() - self.file_offset
            nrows = datasize//rowsize
            self.fobj.seek(self.file_offset)

        return nrows

    def _nativize_dtype(self):
        """
        make sure the dtype is native for text reading

        This will prevent writing text from arrays of a different byte order
        than native: user must change byte order
        """
        descr=self.dtype.descr
        dout=[]
        for d in descr:
            d=list(d)
            # skip over byte order stuff
            d[1] = d[1][1:]
            d=tuple(d)
            dout.append(d)
        self.dtype = numpy.dtype(dout)

    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __repr__(self):
        return _make_repr(self.dtype, self.nrows, self.ncols, self.fobj.mode)

def _make_repr(dtype, nrows, ncols, mode, title='Recfile'):
    topformat=title+"  nrows: %s ncols: %s mode: '%s'\n"
    lines=[]
    line=topformat % (nrows, ncols, mode)
    lines.append(line)

    flines=_get_field_info(dtype, indent='  ')
    lines += flines
           
    lines='\n'.join(lines)

    return lines


def _get_field_info(dtype, indent='  '):

    names=dtype.names

    lines=[]

    nname = 15
    ntype = 6

    # this one is prettier since lines wrap after long names
    format  = indent + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
    long_format = indent + "%-" + str(nname) + "s\n %" + str(len(indent)+nname+ntype) + "s  %s"

    max_pretty_slen = 25
    
    for i in range(len(names)):

        hasfields=False

        n=names[i]

        type = dtype.fields[n][0]

        shape_str = ','.join( str(s) for s in type.shape)

        if type.names is not None:
            ptype = 'rec'
            d=''
            hasfields=True
        else:
            if shape_str != '':
                ptype = type.base.str
                d = 'array[%s]' % shape_str
            else:
                ptype = type.str
                d=''
        
        if len(n) > 15:
            l = long_format % (n,ptype,d)
        else:
            l = format % (n,ptype,d)
        lines.append(l)

        if hasfields:
            recurse_indent = indent + ' '*4
            morelines = _get_field_info(type, indent=recurse_indent)
            lines += morelines

    return lines


class RecfileColumnSubset:
    """

    A class representing a subset of the the columns on disk.  When called with
    .read() or .read(rows=) or slice [ rows ]  the data are read from disk.

    Useful because subsets can be passed around to functions, or chained with a
    row selection.  e.g.

        r=Recfile(...)
        data=r[cols][rows]
    """

    def __init__(self, rf, columns):
        """
        Input is the SFile instance and a list of column names.
        """

        self.recfile = rf
        self.columns = columns

    def read(self, **keys):
        """
        Read the data from disk and return as a numpy array
        """

        c=keys.get('columns',None)
        if c is None:
            keys['columns'] = self.columns
        return self.recfile.read(**keys)

    def __getitem__(self, arg):
        """
        If columns are sent, then the columns will just get reset and
        we'll return a new object

        If rows are sent, they are read and the result returned.
        """

        # we have to unpack the rows if we are reading a subset of the columns
        # because our slice operator only works on whole rows.
        res, isrows, isslice = \
            self.recfile._process_args_as_rows_or_columns(arg, unpack=True)

        if isrows:
            # rows was entered: read all current column subset
            return self.read(rows=res)

        # columns was entered.  Return a new column subset object
        return RecfileColumnSubset(self.rf, self.columns)


    def __repr__(self):
        cols2read = self.recfile._get_cols2read(self.columns)
        descr=[]
        for i in cols2read:
            descr.append(self.recfile.dtype.descr[i])
        dtype=numpy.dtype(descr)
        return _make_repr(dtype, self.recfile.nrows, cols2read.size, 
                          self.recfile.fobj.mode,
                          title='RecfileColumnSubset')




# Testing code

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReadWrite)
    unittest.TextTestRunner(verbosity=2).run(suite)

class TestReadWrite(unittest.TestCase):
    def setUp(self):
 
        nrows=20
        nvec = 2
        ashape=(3,3)
        Sdtype = 'S7'
        # all currently available types, scalar, 1-d and 2-d array columns
        dtype=[('u1scalar','u1'),
               ('i1scalar','i1'),
               ('u2scalar','u2'),
               ('i2scalar','i2'),
               ('u4scalar','u4'),
               ('i4scalar','i4'),
               ('i8scalar','i8'),
               ('f4scalar','f4'),
               ('f8scalar','f8'),

               ('u1vec','u1',nvec),
               ('i1vec','i1',nvec),
               ('u2vec','u2',nvec),
               ('i2vec','i2',nvec),
               ('u4vec','u4',nvec),
               ('i4vec','i4',nvec),
               ('i8vec','i8',nvec),
               ('f4vec','f4',nvec),
               ('f8vec','f8',nvec),
 
               ('u1arr','u1',ashape),
               ('i1arr','i1',ashape),
               ('u2arr','u2',ashape),
               ('i2arr','i2',ashape),
               ('u4arr','u4',ashape),
               ('i4arr','i4',ashape),
               ('i8arr','i8',ashape),
               ('f4arr','f4',ashape),
               ('f8arr','f8',ashape),

               ('Sscalar',Sdtype),
               ('Svec',   Sdtype, nvec),
               ('Sarr',   Sdtype, ashape)]

        dtype2=[('index','i4'),
                ('x','f8'),
                ('y','f8')]

        data=numpy.zeros(nrows, dtype=dtype)

        for t in ['u1','i1','u2','i2','u4','i4','i8','f4','f8']:
            data[t+'scalar'] = 1 + numpy.arange(nrows, dtype=t)
            data[t+'vec'] = 1 + numpy.arange(nrows*nvec,dtype=t).reshape(nrows,nvec)
            arr = 1 + numpy.arange(nrows*ashape[0]*ashape[1],dtype=t)
            data[t+'arr'] = arr.reshape(nrows,ashape[0],ashape[1])

        vid=0
        aid=0
        for row in xrange(nrows):
            data['Sscalar'][row] = 's%02d' % row
            for i in xrange(nvec):
                data['Svec'][row,i] = 'vec%02d' % vid
                vid += 1
            for i in xrange(ashape[0]):
                for j in xrange(ashape[1]):
                    data['Sarr'][row,i,j] = 'arr%03d' % aid
                    aid += 1

        self.data=data

    def testAsciiWriteRead(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values
        """

        delims = {'csv':',',
                  'dat':' ',
                  'colon':':',
                  'tab':'\t'}
        #delims={'csv':','}
        for delim_type in delims:
            try:
                delim=delims[delim_type]
                fname=tempfile.mktemp(prefix='recfile-AsciiWrite-',
                                      suffix='.'+delim_type)
                rw=Recfile(fname,'w+',dtype=self.data.dtype,
                           delim=delims[delim_type])
                rw.write(self.data)

                d = rw.read()
                self.compare_rec(self.data, d, 
                                 "delim %s read/write" % delim_type)


                for f in self.data.dtype.names:
                    d = rw.read(columns=f)
                    self.compare_array(self.data[f], d, "single col read '%s'" % f)

                # now list of columns
                for cols in [['u2scalar','f4vec','Sarr'],
                             ['f8scalar','u2arr','Sscalar']]:
                    d = rw.read(columns=cols)
                    for f in d.dtype.names: 
                        self.compare_array(self.data[f][:], d[f], 
                                           "test column list %s" % f)


                    rows = [1,3]
                    d = rw.read(columns=cols, rows=rows)
                    for f in d.dtype.names: 
                        self.compare_array(self.data[f][rows], d[f], 
                                           "test column list %s row subset" % f)

                rw.close()
            finally:
                if os.path.exists(fname):
                    #pass
                    os.remove(fname)

    def testAsciiSlice(self):
        """
        Test with slice notation
        """

        delims = {'csv':',',
                  'dat':' ',
                  'colon':':',
                  'tab':'\t'}
        #delims={'csv':','}

        for delim_type in delims:
            try:
                delim=delims[delim_type]
                fname=tempfile.mktemp(prefix='recfile-AsciiSlice-',
                                      suffix='.'+delim_type)
                rw=Recfile(fname,'w+',dtype=self.data.dtype,delim=delim)
                rw.write(self.data)

                d = rw[:]
                self.compare_rec(self.data, d, "all slice")

                d = rw[2:9]
                self.compare_rec(self.data[2:9], d, "range")

                d = rw[2:17:3]
                self.compare_rec(self.data[2:17:3], d, "slice step != 1")


                # column subsets with slice notation
                for f in self.data.dtype.names:
                    d = rw[f][:]
                    self.compare_array(self.data[f], d, "single col read '%s'" % f)

                # now list of columns
                for cols in [['u2scalar','f4vec','Sarr'],
                             ['f8scalar','u2arr','Sscalar']]:
                    d = rw[cols][:]
                    for f in d.dtype.names: 
                        self.compare_array(self.data[f][:], d[f], 
                                           "test column list %s" % f)


                    rows = [1,3]
                    d = rw[cols][rows]
                    for f in d.dtype.names: 
                        self.compare_array(self.data[f][rows], d[f], 
                                           "test column list %s row subset" % f)

                rw.close()
            finally:
                if os.path.exists(fname):
                    #pass
                    os.remove(fname)



    def testBinaryWriteRead(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values
        """

        fname=tempfile.mktemp(prefix='recfile-BinaryWrite-',suffix='.rec')
        try:
            rw=Recfile(fname,'w+',dtype=self.data.dtype)
            rw.write(self.data)

            d = rw.read()
            self.compare_rec(self.data, d, "read/write")

            for f in self.data.dtype.names:
                d = rw.read(columns=f)
                self.compare_array(self.data[f], d, "single col read '%s'" % f)

            # now list of columns
            for cols in [['u2scalar','f4vec','Sarr'],
                         ['f8scalar','u2arr','Sscalar']]:
                d = rw.read(columns=cols)
                for f in d.dtype.names: 
                    self.compare_array(self.data[f][:], d[f], 
                                       "test column list %s" % f)


                rows = [1,3]
                d = rw.read(columns=cols, rows=rows)
                for f in d.dtype.names: 
                    self.compare_array(self.data[f][rows], d[f], 
                                       "test column list %s row subset" % f)

            rw.close()
        finally:
            if os.path.exists(fname):
                #pass
                os.remove(fname)


    def testBinarySlice(self):
        """
        Test with slice notation
        """
        fname=tempfile.mktemp(prefix='recfile-BinarySlice-',suffix='.rec')
        try:
            rw=Recfile(fname,'w+',dtype=self.data.dtype)
            rw.write(self.data)

            d = rw[:]
            self.compare_rec(self.data, d, "all slice")

            d = rw[2:9]
            self.compare_rec(self.data[2:9], d, "range")

            d = rw[2:17:3]
            self.compare_rec(self.data[2:17:3], d, "slice step != 1")

            # column subsets with slice notation
            for f in self.data.dtype.names:
                d = rw[f][:]
                self.compare_array(self.data[f], d, "single col read '%s'" % f)

            # now list of columns
            for cols in [['u2scalar','f4vec','Sarr'],
                         ['f8scalar','u2arr','Sscalar']]:
                d = rw[cols][:]
                for f in d.dtype.names: 
                    self.compare_array(self.data[f][:], d[f], 
                                       "test column list %s" % f)


                rows = [1,3]
                d = rw[cols][rows]
                for f in d.dtype.names: 
                    self.compare_array(self.data[f][rows], d[f], 
                                       "test column list %s row subset" % f)



            rw.close()
        finally:
            if os.path.exists(fname):
                #pass
                os.remove(fname)




    def compare_array_tol(self, arr1, arr2, tol, name):
        self.assertEqual(arr1.shape, arr2.shape,
                         "testing arrays '%s' shapes are equal: "
                         "input %s, read: %s" % (name, arr1.shape, arr2.shape))

        adiff = numpy.abs( (arr1-arr2)/float(arr1) )
        maxdiff = adiff.max()
        res=numpy.where(adiff  > tol)
        for i,w in enumerate(res):
            self.assertEqual(w.size,0,
                             "testing array '%s' dim %d are "
                             "equal within tolerance %e, found "
                             "max diff %e" % (name,i,tol,maxdiff))


    def compare_array(self, arr1, arr2, name):
        self.assertEqual(arr1.shape, arr2.shape,
                         "testing arrays '%s' shapes are equal: "
                         "input %s, read: %s" % (name, arr1.shape, arr2.shape))

        res=numpy.where(arr1 != arr2)
        for i,w in enumerate(res):
            self.assertEqual(w.size,0,"testing array '%s' dim %d are equal" % (name,i))

    def compare_rec(self, rec1, rec2, name):
        for f in rec1.dtype.names:
            self.assertEqual(rec1[f].shape, rec2[f].shape,
                             "testing '%s' field '%s' shapes are equal: "
                             "input %s, read: %s" % (name, f,rec1[f].shape, rec2[f].shape))

            res=numpy.where(rec1[f] != rec2[f])
            for w in res:
                self.assertEqual(w.size,0,
                                 "testing '%s'\n\tcolumn %s" % (name,f))

    def compare_rec_subrows(self, rec1, rec2, rows, name):
        for f in rec1.dtype.names:
            self.assertEqual(rec1[f][rows].shape, rec2[f].shape,
                             "testing '%s' field '%s' shapes are equal: "
                             "input %s, read: %s" % (name, f,rec1[f].shape, rec2[f].shape))

            res=numpy.where(rec1[f][rows] != rec2[f])
            for w in res:
                self.assertEqual(w.size,0,"testing column %s" % f)




