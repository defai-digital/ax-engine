import Foundation

/// First-run helpers. Ordering compares download size with unified memory;
/// it is a size comparison, not a fit or qualification claim.
public enum FirstRun {
    /// Catalog entries for a first download: those whose download size is
    /// within this Mac's memory, smallest first, then unknown sizes, then
    /// larger packs.
    public static func ordered(_ catalog: [CatalogModel], memoryBytes: Int64) -> [CatalogModel] {
        func rank(_ item: CatalogModel) -> (Int, Int64) {
            guard let size = item.approxSizeBytes else { return (1, 0) }
            return size <= memoryBytes ? (0, size) : (2, size)
        }
        return catalog.enumerated().sorted { left, right in
            let l = rank(left.element), r = rank(right.element)
            return l == r ? left.offset < right.offset : l < r
        }.map(\.element)
    }

    /// Show the guide when a complete scan found no models at all and it has
    /// not been finished or dismissed. A failed or partial scan never offers
    /// it, so an existing library cannot be mistaken for a first run.
    public static func shouldOffer(completed: Bool, discoveryAttempted: Bool, scanning: Bool, libraryCount: Int,
                                   scanFailed: Bool, serverActive: Bool) -> Bool {
        !completed && discoveryAttempted && !scanning && !scanFailed && libraryCount == 0 && !serverActive
    }
}
