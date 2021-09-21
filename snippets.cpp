const auto container = std::vector<int>{1, 3, 224, 224};

const auto beg = std::begin(container);
const auto end = std::end(container);

for (auto it = beg; it != end; ++it) { std::cout << *it << std::endl; }

for (const auto& elem : container) { std::cout << elem << std::endl; }

std::for_each(beg, end, [](const auto& elem) { std::cout << elem << std::endl; });

const auto sorted = std::is_sorted(beg, end);

/**************************************************************************************************/
const auto container = std::array<int, 4>{1, 3, 224, 224};

const auto beg = std::begin(container);
const auto end = std::end(container);

for (auto it = beg; it != end; ++it) { std::cout << *it << std::endl; }

for (const auto& elem : container) { std::cout << elem << std::endl; }

std::for_each(beg, end, [](const auto& elem) { std::cout << elem << std::endl; });

const auto sorted = std::is_sorted(beg, end);

/**************************************************************************************************/
const auto container = std::set<int>{1, 3, 224, 224};

const auto beg = std::begin(container);
const auto end = std::end(container);

for (auto it = beg; it != end; ++it) { std::cout << *it << std::endl; }

for (const auto& elem : container) { std::cout << elem << std::endl; }

std::for_each(beg, end, [](const auto& elem) { std::cout << elem << std::endl; });

const auto sorted = std::is_sorted(beg, end);

/**************************************************************************************************/
