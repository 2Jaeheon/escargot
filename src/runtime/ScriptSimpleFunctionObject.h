/*
 * Copyright (c) 2022-present Samsung Electronics Co., Ltd
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
 *  USA
 */

#ifndef __EscargotScriptSimpleFunctionObject__
#define __EscargotScriptSimpleFunctionObject__

#include "runtime/ScriptFunctionObject.h"
#include "runtime/Environment.h"
#include "runtime/EnvironmentRecord.h"
#include "runtime/ErrorObject.h"
#include "runtime/VMInstance.h"
#include "interpreter/ByteCode.h"
#include "interpreter/ByteCodeGenerator.h"
#include "interpreter/ByteCodeInterpreter.h"

namespace Escargot {

// ScriptSimpleFunctionObject currently supports only 4 registerFileSize (4, 8, 16, 24)
template <bool isStrict = false, bool shouldClearStack = false, unsigned registerFileSize = 4>
class ScriptSimpleFunctionObject : public ScriptFunctionObject {
    friend class Global;

protected:
    ScriptSimpleFunctionObject() // ctor for reading tag
        : ScriptFunctionObject()
    {
    }

    virtual bool isScriptSimpleFunctionObject() const override
    {
        return true;
    }

    virtual Value call(ExecutionState& state, const Value& thisValue, const size_t argc, Value* argv) override
    {
        CHECK_STACK_OVERFLOW(state);

        ASSERT(codeBlock()->isInterpretedCodeBlock());
        InterpretedCodeBlock* codeBlock = interpretedCodeBlock();

        // prepare ByteCodeBlock if needed
        if (UNLIKELY(codeBlock->byteCodeBlock() == nullptr)) {
            generateByteCodeBlock(state);
        }

        ByteCodeBlock* blk = codeBlock->byteCodeBlock();
        Context* ctx = codeBlock->context();
        const size_t registerSize = blk->m_requiredOperandRegisterNumber;
        const size_t programStart = reinterpret_cast<const size_t>(blk->m_code.data());

#if !defined(NDEBUG)
        const size_t stackStorageSize = codeBlock->totalStackAllocatedVariableSize();
        const size_t literalStorageSize = blk->m_numeralLiteralData.size();
        ASSERT(codeBlock->isStrict() == isStrict);
        ASSERT(blk->m_requiredOperandRegisterNumber + stackStorageSize + literalStorageSize <= registerFileSize);
#endif

        // prepare env, ec
        ASSERT(codeBlock->canAllocateEnvironmentOnStack());
        FunctionEnvironmentRecordOnStack<false, false> record(this);
        LexicalEnvironment lexEnv(&record, outerEnvironment()
#ifndef NDEBUG
                                               ,
                                  false
#endif
        );

        // keep ByteCodeBlock pointer in registerFileBuffer
        char registerFileBuffer[sizeof(Value) * registerFileSize + sizeof(size_t)];
        Value* registerFile = reinterpret_cast<Value*>(registerFileBuffer);
        memcpy(registerFileBuffer + sizeof(Value) * registerFileSize, &blk, sizeof(size_t));

        Value* stackStorage = registerFile + registerSize;

        ExecutionState newState(ctx, &state, &lexEnv, argc, argv, isStrict);
        if (isStrict) {
            stackStorage[0] = thisValue;
        } else {
            if (thisValue.isUndefinedOrNull()) {
                stackStorage[0] = newState.context()->globalObjectProxy();
            } else {
                stackStorage[0] = thisValue.toObject(newState);
            }
        }

#if defined(ENABLE_TCO)
        const Value returnValue = Interpreter::interpret(&newState, blk, programStart, registerFile);
        if (shouldClearStack) {
            clearStack<512>();
        }
        if (UNLIKELY(newState.inTCO())) {
            // callee has been called in tail call, so reset the argument buffer
            memset(ThreadLocal::tcoBuffer(), 0, sizeof(Value) * TCO_ARGUMENT_COUNT_LIMIT);
        }
        return returnValue;
#else
        if (shouldClearStack) {
            const Value returnValue = Interpreter::interpret(&newState, blk, programStart, registerFile);
            clearStack<512>();
            return returnValue;
        } else {
            return Interpreter::interpret(&newState, blk, programStart, registerFile);
        }
#endif
    }

    virtual Value construct(ExecutionState& state, const size_t argc, Value* argv, Object* newTarget) override
    {
        // Assert: Type(newTarget) is Object.
        ASSERT(newTarget->isObject());
        ASSERT(newTarget->isConstructor());
        // Let kind be F’s [[ConstructorKind]] internal slot.
        ASSERT(constructorKind() == ConstructorKind::Base); // this is always `Base` because we define ScriptClassConsturctor::construct

        CHECK_STACK_OVERFLOW(state);

        ASSERT(codeBlock()->isInterpretedCodeBlock());
        InterpretedCodeBlock* codeBlock = interpretedCodeBlock();

        // Let thisArgument be ? OrdinaryCreateFromConstructor(newTarget, "%ObjectPrototype%").
        Object* proto = Object::getPrototypeFromConstructor(state, newTarget, [](ExecutionState& state, Context* constructorRealm) -> Object* {
            return constructorRealm->globalObject()->objectPrototype();
        });

        // Set the [[Prototype]] internal slot of obj to proto.
        Object* thisArgument = new Object(state, proto);
        thisArgument->preparePropertyStorage(codeBlock->constructedObjectPropertyCount());

        // prepare ByteCodeBlock if needed
        if (UNLIKELY(codeBlock->byteCodeBlock() == nullptr)) {
            generateByteCodeBlock(state);
        }

        ByteCodeBlock* blk = codeBlock->byteCodeBlock();
        Context* ctx = codeBlock->context();
        const size_t registerSize = blk->m_requiredOperandRegisterNumber;

#if !defined(NDEBUG)
        const size_t stackStorageSize = codeBlock->totalStackAllocatedVariableSize();
        const size_t literalStorageSize = blk->m_numeralLiteralData.size();
        ASSERT(codeBlock->isStrict() == isStrict);
        ASSERT(blk->m_requiredOperandRegisterNumber + stackStorageSize + literalStorageSize <= registerFileSize);
#endif

        // prepare env, ec
        ASSERT(codeBlock->canAllocateEnvironmentOnStack());
        FunctionEnvironmentRecordOnStack<false, true> record(this);
        LexicalEnvironment lexEnv(&record, outerEnvironment()
#ifndef NDEBUG
                                               ,
                                  false
#endif
        );

        // keep ByteCodeBlock pointer in registerFileBuffer
        char registerFileBuffer[sizeof(Value) * registerFileSize + sizeof(size_t)];
        Value* registerFile = reinterpret_cast<Value*>(registerFileBuffer);
        memcpy(registerFileBuffer + sizeof(Value) * registerFileSize, &blk, sizeof(size_t));
        Value* stackStorage = registerFile + registerSize;

        ExecutionState newState(ctx, &state, &lexEnv, argc, argv, isStrict);
        stackStorage[0] = thisArgument;
        record.setNewTarget(newTarget);

        const Value returnValue = Interpreter::interpret(&newState, blk, reinterpret_cast<const size_t>(blk->m_code.data()), registerFile);
        if (shouldClearStack) {
            clearStack<512>();
        }

        // store pre-allocated storage size later
        codeBlock->setConstructedObjectPropertyCount(thisArgument->ownPropertyCountOnStructure());
        return returnValue.isObject() ? returnValue : thisArgument;
    }
};

COMPILE_ASSERT(sizeof(ScriptSimpleFunctionObject<>) == sizeof(ScriptFunctionObject), "");

} // namespace Escargot

#endif
