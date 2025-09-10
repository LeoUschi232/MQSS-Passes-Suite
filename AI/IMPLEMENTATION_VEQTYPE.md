```C++
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace quake {
class CableType;
class ControlType;
class MeasureType;
class RefType;
class StruqType;
class VeqType;
class WireType;
class StateType;
namespace detail {
struct CableTypeStorage;
} // namespace detail
class CableType : public ::mlir::Type::TypeBase<CableType, mlir::Type, detail::CableTypeStorage> {
public:
  using Base::Base;
  static CableType get(::mlir::MLIRContext *context, std::uint64_t size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"cable"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  std::uint64_t getSize() const;
};
class ControlType : public ::mlir::Type::TypeBase<ControlType, mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"control"};
  }

};
class MeasureType : public ::mlir::Type::TypeBase<MeasureType, mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"measure"};
  }

};
class RefType : public ::mlir::Type::TypeBase<RefType, mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ref"};
  }

};
namespace detail {
struct StruqTypeStorage;
} // namespace detail
class StruqType : public ::mlir::Type::TypeBase<StruqType, mlir::Type, detail::StruqTypeStorage> {
public:
  using Base::Base;
  std::size_t getNumMembers() const { return getMembers().size(); }
  static StruqType get(::mlir::MLIRContext *context, mlir::StringAttr name, ::llvm::ArrayRef<mlir::Type> members);
  static StruqType get(::mlir::MLIRContext *context, llvm::ArrayRef<mlir::Type> members);
  static StruqType get(::mlir::MLIRContext *context, llvm::StringRef name, llvm::ArrayRef<mlir::Type> members);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"struq"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  mlir::StringAttr getName() const;
  ::llvm::ArrayRef<mlir::Type> getMembers() const;
};
namespace detail {
struct VeqTypeStorage;
} // namespace detail
class VeqType : public ::mlir::Type::TypeBase<VeqType, mlir::Type, detail::VeqTypeStorage> {
public:
  using Base::Base;
  static constexpr std::size_t kDynamicSize =
    std::numeric_limits<std::size_t>::max();

  bool hasSpecifiedSize() const { return getSize() != kDynamicSize; }
  static VeqType getUnsized(mlir::MLIRContext *ctx) {
    return VeqType::get(ctx, kDynamicSize);
  }
  static VeqType get(::mlir::MLIRContext *context, std::size_t size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"veq"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  std::size_t getSize() const;
};
class WireType : public ::mlir::Type::TypeBase<WireType, mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"wire"};
  }

};
class StateType : public ::mlir::Type::TypeBase<StateType, mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"state"};
  }

};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::CableType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ControlType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::MeasureType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::RefType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::StruqType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::VeqType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::WireType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::StateType)

#endif  // GET_TYPEDEF_CLASSES
```

```C++
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::quake::CableType,
::quake::ControlType,
::quake::MeasureType,
::quake::RefType,
::quake::StruqType,
::quake::VeqType,
::quake::WireType,
::quake::StateType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::quake::CableType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::CableType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::quake::ControlType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::ControlType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::quake::MeasureType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::MeasureType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::quake::RefType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::RefType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::quake::StruqType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::StruqType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::quake::VeqType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::VeqType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::quake::WireType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::WireType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::quake::StateType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::quake::StateType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)    .Case<::quake::CableType>([&](auto t) {
      printer << ::quake::CableType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::quake::ControlType>([&](auto t) {
      printer << ::quake::ControlType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::quake::MeasureType>([&](auto t) {
      printer << ::quake::MeasureType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::quake::RefType>([&](auto t) {
      printer << ::quake::RefType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::quake::StruqType>([&](auto t) {
      printer << ::quake::StruqType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::quake::VeqType>([&](auto t) {
      printer << ::quake::VeqType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::quake::WireType>([&](auto t) {
      printer << ::quake::WireType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::quake::StateType>([&](auto t) {
      printer << ::quake::StateType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace quake {
namespace detail {
struct CableTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<std::uint64_t>;
  CableTypeStorage(std::uint64_t size) : size(size) {}

  KeyTy getAsKey() const {
    return KeyTy(size);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (size == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CableTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto size = std::get<0>(tblgenKey);
    return new (allocator.allocate<CableTypeStorage>()) CableTypeStorage(size);
  }

  std::uint64_t size;
};
} // namespace detail
CableType CableType::get(::mlir::MLIRContext *context, std::uint64_t size) {
  return Base::get(context, size);
}

::mlir::Type CableType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<std::uint64_t> _result_size;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'size'
  _result_size = ::mlir::FieldParser<std::uint64_t>::parse(odsParser);
  if (::mlir::failed(_result_size)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CableType parameter 'size' which is to be a `std::uint64_t`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_size));
  return CableType::get(odsParser.getContext(),
      std::uint64_t((*_result_size)));
}

void CableType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getSize());
  odsPrinter << ">";
}

std::uint64_t CableType::getSize() const {
  return getImpl()->size;
}

} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::CableType)
namespace quake {
} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::ControlType)
namespace quake {
} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::MeasureType)
namespace quake {
} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::RefType)
namespace quake {
namespace detail {
struct StruqTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::StringAttr, ::llvm::ArrayRef<mlir::Type>>;
  StruqTypeStorage(mlir::StringAttr name, ::llvm::ArrayRef<mlir::Type> members) : name(name), members(members) {}

  KeyTy getAsKey() const {
    return KeyTy(name, members);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey)) && (members == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static StruqTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto name = std::get<0>(tblgenKey);
    auto members = std::get<1>(tblgenKey);
    members = allocator.copyInto(members);
    return new (allocator.allocate<StruqTypeStorage>()) StruqTypeStorage(name, members);
  }

  mlir::StringAttr name;
  ::llvm::ArrayRef<mlir::Type> members;
};
} // namespace detail
StruqType StruqType::get(::mlir::MLIRContext *context, mlir::StringAttr name, ::llvm::ArrayRef<mlir::Type> members) {
  return Base::get(context, name, members);
}

StruqType StruqType::get(::mlir::MLIRContext *context, llvm::ArrayRef<mlir::Type> members) {
  return Base::get(context, mlir::StringAttr{}, members);
}

StruqType StruqType::get(::mlir::MLIRContext *context, llvm::StringRef name, llvm::ArrayRef<mlir::Type> members) {
  return Base::get(context, mlir::StringAttr::get(context, name), members);
}

mlir::StringAttr StruqType::getName() const {
  return getImpl()->name;
}

::llvm::ArrayRef<mlir::Type> StruqType::getMembers() const {
  return getImpl()->members;
}

} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::StruqType)
namespace quake {
namespace detail {
struct VeqTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<std::size_t>;
  VeqTypeStorage(std::size_t size) : size(size) {}

  KeyTy getAsKey() const {
    return KeyTy(size);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (size == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static VeqTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto size = std::get<0>(tblgenKey);
    return new (allocator.allocate<VeqTypeStorage>()) VeqTypeStorage(size);
  }

  std::size_t size;
};
} // namespace detail
VeqType VeqType::get(::mlir::MLIRContext *context, std::size_t size) {
  return Base::get(context, size);
}

std::size_t VeqType::getSize() const {
  return getImpl()->size;
}

} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::VeqType)
namespace quake {
} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::WireType)
namespace quake {
} // namespace quake
MLIR_DEFINE_EXPLICIT_TYPE_ID(::quake::StateType)
namespace quake {

/// Parse a type registered to this dialect.
::mlir::Type QuakeDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void QuakeDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace quake

#endif  // GET_TYPEDEF_CLASSES
```