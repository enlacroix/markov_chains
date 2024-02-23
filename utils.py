def array_to_tex_matrix(arr):
    num_rows, num_cols = arr.shape
    matrix_str = "\\begin{pmatrix}\n"
    for i in range(num_rows):
        row_str = " & ".join(str(val) for val in arr[i])
        matrix_str += row_str
        if i < num_rows - 1:
            matrix_str += " \\\\\n"
        else:
            matrix_str += "\n"
    matrix_str += "\\end{pmatrix}"
    return matrix_str


def find_common_denominator(lst, limit=5000):
    lst = [round(x, 5) for x in lst]
    EPS = 0.3
    for num in range(2, limit + 1):
        if all((abs(int(num * elem) - num * elem) < EPS) and int(num * elem) > 0 for elem in lst):
            return [f'{int(num * elem)}/{num}' for elem in lst]
    return

