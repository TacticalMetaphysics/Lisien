
export function compare_arrays(array1, array2) {
    return array1.map(function (value, index) {
        return value === array2[index]
    })
}


