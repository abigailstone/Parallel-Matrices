//******************************************************
//	MATRIX LIBRARY
//	Basic matrix operations used by matrix mpy and CNN on CPU & GPU.
//******************************************************

#include <vector>
#include <sstream>
#include <cassert>

///////////////////////////////
// Matrix class
// methods to declare a matrix, allocate space, initialize it, do slow
// single-threaded matrix multiply, printing
///////////////////////////////
class Matrix {
    // The private members.
    std::vector<float> data;	
    int _nRows, _nCols;	
    int _log2NColsAlc;	// Always a power of 2, to make index() faster.
    int index (int r, int c) const { 
      return ((r << _log2NColsAlc) | c); 
    }

  public:
    Matrix (int rows, int cols= -1);	// Create a matrix, allocate its storage
    int nRows() const { 
      return (this->_nRows); 
    }

    int nCols() const { 
      return (this->_nCols);
    
    }
    int N() const { 
      assert(_nRows==_nCols); 
      return (_nRows); 
    }

    // Access an element
    float &operator() (int r,int c) {
      return(this->data[this->index(r,c)]);
    }

    float operator() (int r,int c) const {
      return(this->data[this->index(r,c)]);
    }

    bool operator== (const Matrix &other) const;	// Full equality check
    void compare (const Matrix &M2) const;		// Die on first mismatch

    // Initialize a matrix; to I, to random #s in [0,1], or cyclic ints.
    void init_identity();
    void init_random (int max);
    void init_cyclic_order ();
    void init_count_order ();

    std::string row_str(int row) const;	// Print one matrix row to a string.
    std::string str() const;		// Ditto for the entire matrix.

    void conv_naive (const Matrix &array_in, const Matrix &f);// 1 thr, unblocked

    // 1 thread, but blocked.
    void conv_host (const Matrix &in, const Matrix &f, int n_procs, int tile_size);

    // multithreaded & blocked.
    void conv_host (const Matrix &array_in, const Matrix &f, int n_procs);

    void matmul_naive (const Matrix &A, const Matrix &B);	// 1 thread, unblocked
    // 1 thread, but blocked.
    void matmul_host (const Matrix &A, const Matrix &B, int BS);
    // multithreaded & blocked.
    void matmul_host (const Matrix &A, const Matrix &B, int BS, int n_procs);
};
