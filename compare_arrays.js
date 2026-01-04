
export function compare_arrays(array1, array2) {
    return array1.map(function (value, index) {
        return value === array2[index]
    })
}

export function compare_lists(list1, list2) {
    return compare_arrays(Array.from(list1), Array.from(list2))
}


